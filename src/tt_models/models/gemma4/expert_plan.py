"""Host planning and matmul configs for expert-major prefill.

Host planning and matmul configs for expert-major prefill
(moe.FeedForward._experts_prefill).
"""

from __future__ import annotations

import torch
import ttnn

from tt_models.common.config import TILE

# Expert-major prefill rounds its chunk count up to a multiple of this (see
# _plan_chunks).
CHUNK_GRANULE = 16


def matmul_1d_config(
    m_tiles: int,
    n: int,
    k: int,
    grid: tuple[int, int] = (8, 8),
    in0_block_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Configure a one-dimensional matmul for expert prefill."""
    n_tiles = n // TILE
    # The 1D mcast kernel needs every core of a full rectangle to own work.
    for per_core_n in range(1, n_tiles + 1):
        cores = -(-n_tiles // per_core_n)
        shape = next(
            (
                (x, cores // x)
                for x in range(grid[0], 0, -1)
                if cores % x == 0 and cores // x <= grid[1]
            ),
            None,
        )
        if shape:
            gx, gy = shape
            break
    # Large K blocks matter for these DRAM-bound decode shapes: measured in
    # docs/experiments/2026-11-17-decode-round-1.md, down (K=11 tiles) took 154 us at
    # in0_block_w=1 and 92 us at 11; gate_up (K=88) 194 us at 8 and 184 us at 22.
    k_tiles = k // TILE
    in0_block_w = in0_block_w or next(
        w for w in (22, 11, 8, 4, 2, 1) if k_tiles % w == 0
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=m_tiles,
        out_block_w=per_core_n,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def plan_chunks(
    ids: torch.Tensor, weights: torch.Tensor, rows: int, n_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host layout of expert-major prefill for top-k ids/weights [T, k].

    Each expert's (token, slot) pairs fill chunks of `rows` rows; padding rows
    gather token 0 and get weight 0, which zeroes their results. The chunk count rounds
    up to a
    multiple of CHUNK_GRANULE with zero chunks: it depends on the routing, and
    every new count would compile new programs. Returns the gather index into
    the token rows [C * rows], the row weights [C * rows], the expert of every
    chunk [C], and for each (slot j, token t) in slot-major order the row that
    holds its result [k * T].
    """
    tokens, k = ids.shape
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    chunks = (counts + rows - 1) // rows
    real = int(chunks.sum())
    extra = -(-real // CHUNK_GRANULE) * CHUNK_GRANULE - real
    chunk_expert = torch.cat(
        [
            torch.repeat_interleave(torch.arange(n_experts), chunks),
            torch.zeros(extra, dtype=torch.long),
        ]
    )
    # first row of each expert's region, then the row of every sorted pair
    region = torch.cumsum(chunks * rows, 0) - chunks * rows
    sorted_e = flat[order]
    rank = torch.arange(flat.numel()) - (torch.cumsum(counts, 0) - counts)[sorted_e]
    row_of_sorted = region[sorted_e] + rank
    total = (real + extra) * rows
    gather = torch.zeros(total, dtype=torch.int32)
    row_w = torch.zeros(total)
    gather[row_of_sorted] = (order // k).int()
    row_w[row_of_sorted] = weights.flatten()[order].float()
    row_of_pair = torch.empty(flat.numel(), dtype=torch.int32)
    row_of_pair[order] = row_of_sorted.int()
    back = row_of_pair.reshape(tokens, k).T.contiguous().flatten()  # slot-major
    return gather, row_w, chunk_expert, back
