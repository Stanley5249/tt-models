"""Shared helpers: mesh mappers, cached weight upload, compute configs."""

from collections.abc import Callable
from pathlib import Path

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32


class WeightLoader:
    """Upload weights through a tensorbin cache so later loads skip host conversion.

    `build` is only called on a cache miss, so a warm start never reads the
    checkpoint for that tensor.
    """

    def __init__(self, mesh: ttnn.MeshDevice, cache_dir: Path):
        self.mesh = mesh
        self.cache_dir = Path(cache_dir)
        self.tp = mesh.get_num_devices()

    def _cached(self, name: str) -> bool:
        return any(self.cache_dir.glob(f"{name}_dtype_*.tensorbin"))

    def put(
        self,
        name: str,
        build: Callable[[], torch.Tensor],
        dtype=ttnn.bfloat16,
        shard_dim: int | None = None,
        layout=ttnn.TILE_LAYOUT,
        memory_config=DRAM,
    ) -> ttnn.Tensor:
        if shard_dim is None:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        else:
            mapper = ttnn.ShardTensorToMesh(self.mesh, dim=shard_dim)
        tensor = None if self._cached(name) else build()
        return ttnn.as_tensor(
            tensor,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=mapper,
            memory_config=memory_config,
            cache_file_name=str(self.cache_dir / name),
        )


def compute_config(fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True, approx=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


HIFI4 = compute_config(ttnn.MathFidelity.HiFi4)
HIFI2 = compute_config(ttnn.MathFidelity.HiFi2)
LOFI = compute_config(ttnn.MathFidelity.LoFi)


def dram_sharded(mesh: ttnn.MeshDevice, k: int, n: int) -> ttnn.MemoryConfig:
    """Width-sharded DRAM config for a [k, n] decode weight, one shard per DRAM bank."""
    banks = mesh.dram_grid_size().x
    padded = -(-n // (TILE * banks)) * TILE * banks
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def width_sharded_l1(width: int, cores: int = 8) -> ttnn.MemoryConfig:
    """[32, width] width-sharded over cores (0, 0)..(cores - 1, 0)."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, width // cores),
        core_grid=ttnn.CoreGrid(x=cores, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )


def norm_weight(w: torch.Tensor) -> torch.Tensor:
    """RMSNorm weights go to device as ROW_MAJOR [1, 1, dim/32, 32]."""
    return w.float().reshape(1, 1, -1, TILE)


class _Semaphores:
    """Double-buffered global semaphores for the async all-gather, as tt_transformers' TT_CCL."""

    def __init__(self, mesh: ttnn.MeshDevice):
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.gather = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(2)]
        self.barrier = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        self.turn = 0

    def next(self):
        self.turn ^= 1
        return self.gather[self.turn], self.barrier[self.turn]


_SEMAPHORES: dict[int, _Semaphores] = {}


def init_ccl(mesh: ttnn.MeshDevice):
    """Allocate the CCL semaphores; call before any trace capture (see Gemma4TT._prepare_traces)."""
    if id(mesh) not in _SEMAPHORES:
        _SEMAPHORES[id(mesh)] = _Semaphores(mesh)


def all_reduce(x: ttnn.Tensor) -> ttnn.Tensor:
    """Sum x over the two chips: gather both partials, then add them locally.

    all_gather_async + fast_reduce_nc takes 34 us on a [32, 2816] decode row and
    58 us on [32, 5632], against 43 and 71 us for ttnn.all_reduce's reduce-scatter
    plus all-gather (scripts/bench_ccl_async.py). Prefill takes the same path in
    DRAM: ttnn.all_reduce caches semaphores and buffers per new shape, and one
    cached after a decode trace capture is overwritten by the trace's replays.
    """
    mesh = x.device()
    shape = x.shape
    mem = L1 if shape[2] <= TILE else DRAM
    gather_sems, barrier = _SEMAPHORES[id(mesh)].next()
    g = ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=1,
        multi_device_global_semaphore=gather_sems,
        num_links=1,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=mem,
        barrier_semaphore=barrier,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    x.deallocate(True)
    # Gather on dim 1: fast_reduce_nc over dim 0 summed only the first row of a [2, 1, 32, W] tile correctly.
    out = ttnn.experimental.fast_reduce_nc(g, dims=[1], compute_kernel_config=HIFI4, memory_config=mem)
    g.deallocate(True)
    if out.shape != shape:  # the gather drops the logical row count; restoring it is metadata only
        out = ttnn.reshape(out, shape, out.padded_shape)
    return out


_NORM_SHARD = {}


def rms_norm(x: ttnn.Tensor, weight, eps: float) -> ttnn.Tensor:
    """RMSNorm over the last dim of [1, 1, T, H].

    For decode (T <= 32) the interleaved op runs on few cores (63 us on
    [32, 2816]); width-sharding over 8 cores is 18.6 us with both conversions
    (scripts/bench_norm.py), so small inputs take that path.
    """
    rows, width = x.shape[2], x.shape[3]
    if rows > TILE or (width // TILE) % 8:
        return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=DRAM, compute_kernel_config=HIFI4)
    if width not in _NORM_SHARD:
        per = width // TILE // 8
        shard = ttnn.create_sharded_memory_config(
            shape=(TILE, per * TILE),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            subblock_w=next(w for w in (4, 2, 1) if per % w == 0),
            block_h=1,
            block_w=per,
            inplace=False,
        )
        _NORM_SHARD[width] = (shard, cfg)
    shard, cfg = _NORM_SHARD[width]
    xs = ttnn.to_memory_config(x, shard)
    y = ttnn.rms_norm(xs, weight=weight, epsilon=eps, program_config=cfg, memory_config=shard, compute_kernel_config=HIFI4)
    xs.deallocate(True)
    out = ttnn.to_memory_config(y, DRAM)
    y.deallocate(True)
    return out


def all_gather_compact(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, 1, 32, W] compact partials -> [1, 2, 32, W] with both chips' (boundary.py); 10 us for 3 tiles."""
    gather_sems, barrier = _SEMAPHORES[id(x.device())].next()
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=1,
        multi_device_global_semaphore=gather_sems,
        num_links=1,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=L1,
        barrier_semaphore=barrier,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
