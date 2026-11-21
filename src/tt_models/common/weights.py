"""Weight upload through a tensorbin cache."""

from collections.abc import Callable
from pathlib import Path

import torch
import ttnn

from tt_models.common.config import DRAM, TILE


class WeightLoader:
    """Upload weights through a tensorbin cache so later loads skip host conversion.

    `build` is only called on a cache miss, so a warm start never reads the
    checkpoint for that tensor.
    """

    def __init__(self, mesh: ttnn.MeshDevice, cache_dir: Path):
        """Set up the weight cache for a mesh and checkpoint."""
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
        """Load a cached tensor or build and upload it."""
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


def norm_weight(w: torch.Tensor) -> torch.Tensor:
    """RMSNorm weights go to device as ROW_MAJOR [1, 1, dim/32, 32]."""
    return w.float().reshape(1, 1, -1, TILE)
