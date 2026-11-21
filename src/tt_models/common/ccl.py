"""Cross-chip collectives on the N300 1x2 mesh.

Cross-chip collectives on the N300 1x2 mesh (async all-gather with double-buffered
semaphores).
"""

from __future__ import annotations

import ttnn

from tt_models.common.config import DRAM, HIFI4, L1, TILE
from tt_models.common.mesh import per_mesh_cache


class _Semaphores:
    """Double-buffered semaphores for async all-gather.

    Double-buffered global semaphores for the async all-gather, as tt_transformers'
    TT_CCL.
    """

    def __init__(self, mesh: ttnn.MeshDevice) -> None:
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1)
                )
            }
        )
        self.gather = [
            [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
            for _ in range(2)
        ]
        self.barrier = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        self.turn = 0

    def next(
        self,
    ) -> tuple[
        list[ttnn._ttnn.global_semaphore.GlobalSemaphore],
        ttnn._ttnn.global_semaphore.GlobalSemaphore,
    ]:
        self.turn ^= 1
        return self.gather[self.turn], self.barrier[self.turn]


_SEMAPHORES: dict[int, _Semaphores] = per_mesh_cache()


def init_ccl(mesh: ttnn.MeshDevice) -> None:
    """Allocate CCL semaphores before trace capture.

    Allocate the CCL semaphores; call before any trace capture (see
    Gemma4TT._prepare_traces).
    """
    if id(mesh) not in _SEMAPHORES:
        _SEMAPHORES[id(mesh)] = _Semaphores(mesh)


def all_reduce(x: ttnn.Tensor) -> ttnn.Tensor:
    """Sum x over the two chips: gather both partials, then add them locally.

    all_gather_async + fast_reduce_nc takes 34 us on a [32, 2816] decode row and
    58 us on [32, 5632], against 43 and 71 us for ttnn.all_reduce's reduce-scatter
    plus all-gather (docs/experiments/2026-11-19-decode-round-2.md). Prefill takes the
    same path in
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
    # Gather on dim 1: fast_reduce_nc over dim 0 summed only the first row of a [2, 1,
    # 32, W] tile correctly.
    out = ttnn.experimental.fast_reduce_nc(
        g, dims=[1], compute_kernel_config=HIFI4, memory_config=mem
    )
    g.deallocate(True)
    if (
        out.shape != shape
    ):  # the gather drops the logical row count; restoring it is metadata only
        out = ttnn.reshape(out, shape, out.padded_shape)
    return out


def all_gather_compact(x: ttnn.Tensor) -> ttnn.Tensor:
    """Gather compact partial sums from both chips.

    [1, 1, 32, W] compact partials -> [1, 2, 32, W] with both chips' (boundary.py);
    10 us for 3 tiles.
    """
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
