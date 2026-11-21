"""Compact layout of decode hidden vectors (kernels/compact.hpp).

A hidden vector's values as FP32 chunks of 16, 64 chunks per [32, 32] FP32 tile, so 2816 values are 3
dense tiles instead of one row of 88 mostly empty ones. Vector v of a compact tensor starts at chunk
v * 64 * compact_tiles(H); padding chunks are zero.
"""

import torch
import ttnn

from tt_models.common.config import TILE

CHUNK = 16
CHUNKS_PER_TILE = 64


def compact_tiles(hidden: int) -> int:
    return -(-hidden // CHUNK // CHUNKS_PER_TILE)


def to_compact(v: torch.Tensor) -> torch.Tensor:
    """[n, H] -> [32, n * V * 32] float32 whose TILE layout puts chunk c of vector i at byte 64 c of its V tiles."""
    n, h = v.shape
    tiles = compact_tiles(h)
    chunks = torch.zeros(n, tiles * CHUNKS_PER_TILE, CHUNK)
    chunks[:, : h // CHUNK] = v.float().reshape(n, -1, CHUNK)
    # chunk q of a tile = face q // 16, face row q % 16 -> tile row (face // 2) * 16 + r, cols (face % 2) * 16 + j
    c = chunks.reshape(
        n, tiles, 2, 2, 16, CHUNK
    )  # [n, tile, face_row_half, face_col_half, r, j]
    return c.permute(2, 4, 0, 1, 3, 5).reshape(TILE, n * tiles * TILE)


def from_compact(t: torch.Tensor, n: int, h: int) -> torch.Tensor:
    """Inverse of to_compact: [32, n * V * 32] -> [n, h]."""
    tiles = compact_tiles(h)
    c = t.float().reshape(2, 16, n, tiles, 2, CHUNK).permute(2, 3, 0, 4, 1, 5)
    return c.reshape(n, tiles * CHUNKS_PER_TILE * CHUNK)[:, :h]


def compact_tensor(
    mesh, v: torch.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    """Host vectors [n, H] -> device compact FP32 [1, 1, 32, n * V * 32], replicated."""
    return ttnn.from_torch(
        to_compact(v)[None, None],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=memory_config,
    )
