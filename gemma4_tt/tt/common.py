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


def norm_weight(w: torch.Tensor) -> torch.Tensor:
    """RMSNorm weights go to device as ROW_MAJOR [1, 1, dim/32, 32]."""
    return w.float().reshape(1, 1, -1, TILE)


def all_reduce(x: ttnn.Tensor) -> ttnn.Tensor:
    out = ttnn.all_reduce(x, cluster_axis=1, memory_config=DRAM)
    x.deallocate(True)
    return out
