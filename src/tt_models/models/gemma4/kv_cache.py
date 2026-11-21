"""Paged KV cache in vLLM's hybrid layout.

vLLM pages every attention layer: the 25 sliding layers as SlidingWindowSpec
(window 1024) and the 5 global layers as FullAttentionSpec. Its hybrid
manager splits them into KV-cache groups of 5 layers, 5 sliding groups and
1 global group, each with its own page table. A block id belongs to one group
at a time, so the i-th layer of every group shares one K and one V buffer.

The shared buffers need equal bytes per block in every view. vLLM gets that
by giving the global layers 128-token blocks of 1 KV head x 512 per chip,
against 64-token sliding blocks of 4 heads x 256. A layer whose view differs
from the buffer's declared shape passes its geometry to the paged ops.

Page tables travel as one int32 tensor [groups, rows, width] per step. The
sliding groups free blocks that leave the window (vLLM's null block 0 takes
their place), and prefix caching keeps freed blocks until they are evicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import ttnn

from tt_models.common.config import DRAM

if TYPE_CHECKING:
    from tt_models.models.gemma4.attention import AttnShape

SLIDING = "sliding_attention"


def kv_groups(layer_types: list[str]) -> list[int]:
    """The KV-cache group of every layer, as vLLM's hybrid manager forms them.

    Mirrors vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size:
    layers bucket by type in order of first appearance, the group size is the
    smallest bucket, and bucket layers[i::n] form the bucket's i-th group. The
    adapter checks the result against the page tables vLLM sends.
    """
    buckets: dict[str, list[int]] = {}
    for i, kind in enumerate(layer_types):
        buckets.setdefault(kind, []).append(i)
    size = min(len(b) for b in buckets.values())
    if max(len(b) for b in buckets.values()) < size * 1.5:
        size = max(len(b) for b in buckets.values())
    group_of = [0] * len(layer_types)
    first = 0
    for layers in buckets.values():
        n = -(-len(layers) // size)
        for g in range(n):
            for i in layers[g::n]:
                group_of[i] = first + g
        first += n
    return group_of


@dataclass(frozen=True, eq=False)
class LayerKv:
    """One layer's view of a shared K/V buffer pair."""

    k: ttnn.Tensor
    v: ttnn.Tensor
    group: int
    block_size: int
    n_kv: int  # KV heads per chip in this view
    geometry: (
        ttnn.PagedCacheGeometryOverride | None
    )  # None when the view is the buffer's own shape

    def update_args(self) -> dict[str, int]:
        """Return the cache and metadata arguments for a decode update."""
        return (
            {}
            if self.geometry is None
            else {"block_size": self.block_size, "num_kv_heads": self.n_kv}
        )

    def fill_args(self) -> dict[str, int]:
        """Return the cache and page table arguments for a prefill fill."""
        return {} if self.geometry is None else {"block_size": self.block_size}

    def sdpa_args(self) -> dict[str, int]:
        """Return the cache and page table arguments for paged attention."""
        return {} if self.geometry is None else {"paged_cache_geometry": self.geometry}


class KvCache:
    """The device KV buffers and every layer's view of them."""

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        layer_types: list[str],
        shapes: list[tuple[int, int, int, int]],
        buffer_of: list[int],
        group_of: list[int],
        dtype: ttnn.DataType,
    ) -> None:
        """Allocate the paged KV cache for all layers.

        shapes: per layer (num_blocks, KV heads per chip, block_size, head_dim);
        buffer_of: the shared buffer of every layer; group_of: its KV-cache group.
        """
        self.num_blocks = shapes[0][0]
        self.groups = max(group_of) + 1
        self.sliding = [False] * self.groups
        for kind, g in zip(layer_types, group_of):
            self.sliding[g] = kind == SLIDING
        buffers: dict[
            int, tuple[ttnn.Tensor, ttnn.Tensor, tuple[int, int, int, int]]
        ] = {}
        self.layers: list[LayerKv] = []
        self.block_sizes = [0] * self.groups
        for shape, buf, group in zip(shapes, buffer_of, group_of):
            if buf not in buffers:
                pair = [
                    ttnn.zeros(
                        shape,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=DRAM,
                    )
                    for _ in range(2)
                ]
                buffers[buf] = (*pair, shape)
            k, v, own = buffers[buf]
            _, n_kv, block, head_dim = shape
            if n_kv * block * head_dim != own[1] * own[2] * own[3]:
                raise ValueError(
                    f"layer view {shape} and its shared buffer {own} differ in "
                    "elements per block"
                )
            same = shape == own
            geometry = (
                None
                if same
                else ttnn.PagedCacheGeometryOverride(
                    block_size=block, num_kv_heads=n_kv
                )
            )
            self.layers.append(LayerKv(k, v, group, block, n_kv, geometry))
            self.block_sizes[group] = block
        # vLLM schedules on the lcm of the block sizes: prefix hits and chunk ends fall
        # on it.
        self.grid = math.lcm(*self.block_sizes)

    @classmethod
    def standalone(
        cls,
        mesh: ttnn.MeshDevice,
        layer_types: list[str],
        attn_shapes: list[AttnShape],
        block_size: int,
        dtype: ttnn.DataType,
        rows: int,
        tokens: int,
        ring_tokens: int,
    ) -> KvCache:
        """Allocate a standalone cache using vLLM's layout.

        The layout vLLM builds, sized for `rows` sequences of `tokens` positions,
        for scripts and tests that run without vLLM (see tables()).

        attn_shapes: every layer's AttnShape. The largest per-token view keeps
        block_size; the others scale their blocks up to the same bytes.
        """
        group_of = kv_groups(layer_types)
        per_token = [s.n_kv * s.head_dim for s in attn_shapes]
        widest = max(per_token)
        block_of = [block_size * widest // t for t in per_token]
        groups = max(group_of) + 1
        block_sizes, sliding = [0] * groups, [False] * groups
        for g, block, kind in zip(group_of, block_of, layer_types):
            block_sizes[g], sliding[g] = block, kind == SLIDING
        num_blocks = 1 + rows * sum(
            _own_blocks(b, tokens, ring_tokens, sliding=s)
            for b, s in zip(block_sizes, sliding)
        )
        shapes = [
            (num_blocks, s.n_kv, block, s.head_dim)
            for s, block in zip(attn_shapes, block_of)
        ]
        position: dict[int, int] = {}
        buffer_of = []
        for g in group_of:  # the i-th layer of every group shares buffer i
            buffer_of.append(position.get(g, 0))
            position[g] = position.get(g, 0) + 1
        return cls(mesh, layer_types, shapes, buffer_of, group_of, dtype)

    def tables(
        self, rows: int, tokens: int, width: int, ring_tokens: int
    ) -> torch.Tensor:
        """Build page tables for scripts and warmup.

        Page tables [groups, rows, width] for scripts and warmup: every row gets
        `tokens` positions in blocks of its own, after the null block 0.

        Sliding groups cycle through enough blocks for `ring_tokens` positions (the
        window plus one prefill chunk), because only the window is read again.
        """
        out = torch.zeros(self.groups, rows, width, dtype=torch.int32)
        next_block = 1
        for g, block in enumerate(self.block_sizes):
            n = -(-tokens // block)
            own = _own_blocks(block, tokens, ring_tokens, sliding=self.sliding[g])
            for r in range(rows):
                if next_block + own > self.num_blocks or n > width:
                    raise ValueError(
                        f"{rows} rows of {tokens} tokens do not fit {self.num_blocks} "
                        f"blocks, {width} per row"
                    )
                blocks = torch.arange(next_block, next_block + own, dtype=torch.int32)
                out[g, r, :n] = blocks[torch.arange(n) % own]
                next_block += own
        return out


def _own_blocks(block: int, tokens: int, ring_tokens: int, *, sliding: bool) -> int:
    n = -(-tokens // block)
    return min(n, -(-ring_tokens // block)) if sliding else n
