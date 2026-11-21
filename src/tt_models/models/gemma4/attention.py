"""Gemma 4 attention with heads split over the two N300 chips (TP=2).

Sliding layers: 16 Q heads, 8 KV heads, head_dim 256, window 1024.
Global layers: 16 Q heads, 2 KV heads, head_dim 512, K reused as V
(attention_k_eq_v), partial ("proportional") RoPE.

Each chip owns 8 Q heads and the KV heads they read, so attention needs no
communication until the output projection, whose partial sums are all-reduced.

KV caches are paged in vLLM's hybrid layout (kv_cache.py): every layer reads
its group's page table at absolute positions, through its view of a shared
buffer. A prefill chunk at position 0 attends over its own K/V; a later chunk
(chunked prefill, a prefix-cache hit) attends over the paged cache, which then
holds the earlier positions and the chunk itself.
"""

from dataclasses import dataclass

import torch
import ttnn

from tt_models.common.ccl import all_reduce
from tt_models.common.config import DRAM, HIFI2, HIFI4, TILE, dram_sharded, width_sharded_l1
from tt_models.common.weights import WeightLoader, norm_weight
from tt_models.models.gemma4.checkpoint import attention_dims
from tt_models.models.gemma4.kv_cache import LayerKv
from tt_models.models.gemma4.qk_rope import QkRope

# Chunked prefill attention needs its chunk start on a multiple of the SDPA q and
# k chunks, which must also cover every KV block size (64 sliding, 128 global).
CHUNK_ALIGN = 128


@dataclass(frozen=True)
class AttnShape:
    sliding: bool
    head_dim: int
    n_q: int  # per chip
    n_kv: int  # per chip
    window: int | None


def attn_shape(cfg, layer_idx: int, tp: int) -> AttnShape:
    kind = cfg.layer_types[layer_idx]
    head_dim, n_kv = attention_dims(cfg)[kind]
    sliding = kind == "sliding_attention"
    return AttnShape(sliding, head_dim, cfg.num_attention_heads // tp, n_kv // tp, cfg.sliding_window if sliding else None)


@dataclass
class PrefillChunk:
    """Positions [start, start + rows) of one sequence, as padded rows of x.

    start is a multiple of every block size and of the SDPA chunk. tables holds
    each group's page table of the sequence (None for a chunk at 0, which reads
    no cache), fill_tables the same tables from the block holding `start`.
    """

    start: int
    rows: int
    start_tensor: ttnn.Tensor | None  # start on the device, so new offsets reuse the compiled programs
    tables: list[ttnn.Tensor | None]
    fill_tables: list[ttnn.Tensor]


_QK_ROPE = {}


def qk_rope(mesh, width: int, head_dim: int, eps: float) -> QkRope:
    key = (id(mesh), width, head_dim)
    if key not in _QK_ROPE:
        _QK_ROPE[key] = QkRope(mesh, width, head_dim, eps)
    return _QK_ROPE[key]


_ROTATE = {}


def rotate_half(x: ttnn.Tensor, memory_config=DRAM) -> ttnn.Tensor:
    """cat(-x2, x1) as one matmul with a constant +-1 matrix (exact in BF16 with fp32 accumulation)."""
    hd = x.shape[-1]
    key = (hd, id(x.device()))
    if key not in _ROTATE:
        half = hd // 2
        r = torch.zeros(hd, hd)
        r[torch.arange(half) + half, torch.arange(half)] = -1.0
        r[torch.arange(half), torch.arange(half) + half] = 1.0
        mesh = x.device()
        _ROTATE[key] = ttnn.from_torch(
            r[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
    return ttnn.matmul(x, _ROTATE[key], compute_kernel_config=HIFI4, memory_config=memory_config, dtype=x.dtype)


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config=DRAM, dtype=None) -> ttnn.Tensor:
    """x * cos + rotate_half(x) * sin; intermediates keep x's dtype, the sum is `dtype`."""
    rot = rotate_half(x, memory_config)
    a = ttnn.mul(x, cos, memory_config=memory_config)
    b = ttnn.mul(rot, sin, memory_config=memory_config)
    rot.deallocate(True)
    out = ttnn.add(a, b, memory_config=memory_config, dtype=dtype)
    a.deallocate(True)
    b.deallocate(True)
    return out


def head_norm(x: ttnn.Tensor, weight, eps: float, memory_config=DRAM) -> ttnn.Tensor:
    """RMSNorm over head_dim of a [1, H, S, D] or [1, B, H, D] tensor."""
    shape = list(x.shape)
    flat = ttnn.reshape(x, (1, 1, shape[1] * shape[2], shape[3]))
    out = ttnn.rms_norm(flat, weight=weight, epsilon=eps, memory_config=memory_config, compute_kernel_config=HIFI4)
    return ttnn.reshape(out, shape)


class Attention:
    def __init__(self, loader: WeightLoader, ckpt, cfg, layer_idx: int, dtype=ttnn.bfloat8_b):
        tp = loader.tp
        self.cfg = cfg
        self.shape = s = attn_shape(cfg, layer_idx, tp)
        self.eps = cfg.rms_norm_eps
        p = f"layers.{layer_idx}.self_attn."

        def wqkv():
            q = ckpt.get(p + "q_proj.weight")  # [nq*hd, H]
            k = ckpt.get(p + "k_proj.weight")
            v = k if not s.sliding else ckpt.get(p + "v_proj.weight")
            qd, kd = s.n_q * s.head_dim, s.n_kv * s.head_dim
            chips = [torch.cat([q[c * qd : (c + 1) * qd], k[c * kd : (c + 1) * kd], v[c * kd : (c + 1) * kd]]) for c in range(tp)]
            return torch.cat(chips).T.contiguous()[None, None]  # [1, 1, H, tp*(qd+2kd)]

        def wo():
            return ckpt.get(p + "o_proj.weight").T.contiguous()[None, None]  # [1, 1, nq*hd, H]

        self.wqkv = loader.put(f"L{layer_idx}.wqkv", wqkv, dtype, shard_dim=-1)
        self.wo = loader.put(f"L{layer_idx}.wo", wo, dtype, shard_dim=-2)
        # Decode reads wo DRAM-sharded, fed by the concat-heads output, which is
        # already width-sharded one head per core: 47 -> 34 us (sliding), 86 -> 62 us
        # (global) in scripts/bench_dram_sharded.py. Prefill keeps the interleaved copy.
        k_local, hidden = s.n_q * s.head_dim, cfg.hidden_size
        # A cached tensor keeps the memory layout it was saved with, so the copy has its own name.
        self.wo_dram = loader.put(f"L{layer_idx}.wo_dram", wo, dtype, shard_dim=-2, memory_config=dram_sharded(loader.mesh, k_local, hidden))
        self.wo_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k_local // TILE // s.n_q, per_core_M=1, per_core_N=hidden // TILE // s.n_q, fused_activation=None
        )
        self.wo_out = width_sharded_l1(hidden, s.n_q)
        self.wo_cores = [ttnn.CoreCoord(i, 0) for i in range(s.n_q)]  # the wo_out shard cores
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.q_norm = loader.put(f"L{layer_idx}.q_norm", lambda: norm_weight(ckpt.get(p + "q_norm.weight")), layout=rm)
        self.k_norm = loader.put(f"L{layer_idx}.k_norm", lambda: norm_weight(ckpt.get(p + "k_norm.weight")), layout=rm)
        # Decode normalizes all heads of the fused QKV row at once, so the head
        # weights become one [B * heads, hd] multiplier: q_norm rows, k_norm rows,
        # and ones for V (weightless norm). Built per batch size on first use.
        self._head_w = (ckpt.get(p + "q_norm.weight").float(), ckpt.get(p + "k_norm.weight").float())
        self._head_w_dev = {}
        self.mesh = loader.mesh
        # batch-1 decode: fused q/k norm + RoPE (boundary.QkRope) with [w | w permuted] per value
        self.qk_rope = qk_rope(self.mesh, self.heads * s.head_dim, s.head_dim, self.eps)
        q, k = self._head_w
        w_row = torch.cat([q.repeat(s.n_q), k.repeat(s.n_kv), torch.ones(s.n_kv * s.head_dim)])
        self.rope_w = self.qk_rope.weights(w_row)

    # -- prefill -----------------------------------------------------------

    def prefill(self, x: ttnn.Tensor, rope, kv: LayerKv, chunk: PrefillChunk) -> ttnn.Tensor:
        """x [1, 1, S, H] replicated, the chunk's rows. Returns all-reduced attention output [1, 1, S, H].

        rope: (cos, sin) [1, 1, S, hd] for this layer type at the chunk's positions.
        """
        s = self.shape
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=HIFI2, memory_config=DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=s.n_q, num_kv_heads=s.n_kv, transpose_k_heads=False, memory_config=DRAM
        )
        xqkv.deallocate(True)
        q = self._norm_rope(q, self.q_norm, rope)
        k = self._norm_rope(k, self.k_norm, rope)
        v = head_norm(v, None, self.eps)
        self._fill(k, v, kv, chunk)
        window = {} if s.window is None else {"sliding_window_size": s.window}
        if chunk.start == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                program_config=self._sdpa_prefill_config(q.shape[2]),
                compute_kernel_config=HIFI4,
                memory_config=DRAM,
                **window,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                kv.k,
                kv.v,
                chunk.tables[kv.group],
                chunk_start_idx_tensor=chunk.start_tensor,
                scale=1.0,
                program_config=self._sdpa_chunked_config(),
                compute_kernel_config=HIFI4,
                memory_config=DRAM,
                **window,
                **kv.sdpa_args(),
            )
        for t in (q, k, v):
            t.deallocate(True)
        out = ttnn.experimental.nlp_concat_heads(attn, memory_config=DRAM)
        attn.deallocate(True)
        out2 = ttnn.linear(out, self.wo, compute_kernel_config=HIFI2, memory_config=DRAM)
        out.deallocate(True)
        return all_reduce(out2)

    @staticmethod
    def _fill(k: ttnn.Tensor, v: ttnn.Tensor, kv: LayerKv, chunk: PrefillChunk):
        """Write the chunk's K/V rows up to the last tile holding a real position, which
        lies in the block of the chunk's last position: padding never reaches a block
        vLLM did not allocate (the plugin pads tables with the null block 0)."""
        live = -(-chunk.rows // TILE) * TILE
        table = chunk.fill_tables[kv.group]
        for t, cache in ((k, kv.k), (v, kv.v)):
            # A slice covering the whole tensor returns the input itself; never free k/v.
            rows = t if live == t.shape[2] else ttnn.slice(t, [0, 0, 0, 0], [1, t.shape[1], live, t.shape[3]])
            t8 = ttnn.typecast(rows, cache.dtype)
            if rows is not t:
                rows.deallocate(True)
            ttnn.experimental.paged_fill_cache(cache, t8, table, batch_idx=0, **kv.fill_args())
            t8.deallocate(True)

    def _norm_rope(self, x, weight, rope):
        normed = head_norm(x, weight, self.eps)
        x.deallocate(True)
        out = apply_rope(normed, *rope)
        normed.deallocate(True)
        return out

    def _sdpa_prefill_config(self, seq_len: int):
        # From models/demos/gemma4: q=k=256 overflows L1 at head_dim 256, and
        # head_dim 512 needs an (8, 4) grid with 128 chunks.
        def fit(target):
            return next(c for c in (target, 128, 64, 32) if c <= target and seq_len % c == 0)

        if self.shape.head_dim >= 512:
            grid, q_chunk, k_chunk = (8, 4), fit(128), fit(128)
        else:
            grid, q_chunk, k_chunk = (8, 8), fit(256), fit(128)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
        )

    def _sdpa_chunked_config(self):
        """Chunked SDPA over the paged cache; the chunk start must be a multiple of both chunk sizes."""
        grid = (8, 4) if self.shape.head_dim >= 512 else (8, 8)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=CHUNK_ALIGN, k_chunk_size=CHUNK_ALIGN, exp_approx_mode=False
        )

    # -- decode ------------------------------------------------------------

    @property
    def heads(self) -> int:
        """Heads per chip in the fused QKV row: Q, then K, then V."""
        return self.shape.n_q + 2 * self.shape.n_kv

    def head_weights(self, batch: int) -> ttnn.Tensor:
        if batch not in self._head_w_dev:
            q, k = self._head_w
            s = self.shape
            per_user = torch.cat([q.expand(s.n_q, -1), k.expand(s.n_kv, -1), torch.ones(s.n_kv, s.head_dim)])
            self._head_w_dev[batch] = ttnn.from_torch(
                per_user.repeat(batch, 1)[None, None],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=DRAM,
            )
        return self._head_w_dev[batch]

    def decode_compact(self, x: ttnn.Tensor, rope, kv: LayerKv, page_table, cur_pos, boundary) -> ttnn.Tensor:
        """1-4 decode tokens: x is the normed row-form input [1, 1, rows, H]. Returns this chip's o_proj
        partial sums in compact FP32 form (boundary.py, one vector per row), packed straight from the
        width-sharded matmul output. rope: the batch-1 compact tables or the (cos, sin) rows (decode)."""
        rows = x.shape[2]
        heads = self._heads_decode(x, rope, kv, page_table, cur_pos)
        out_sh = ttnn.linear(
            heads, self.wo_dram, program_config=self.wo_cfg, compute_kernel_config=HIFI2, memory_config=self.wo_out, dtype=ttnn.float32
        )
        heads.deallocate(True)
        out = boundary.compact(rows)
        boundary.pack(out_sh, out, self.wo_cores, rows=rows)
        out_sh.deallocate(True)
        return out

    def _heads_decode(self, x, rope, kv: LayerKv, page_table, cur_pos):
        """QKV, q/k norm and RoPE, cache update and SDPA: the concatenated heads, one head per core."""
        s = self.shape
        batch = x.shape[2]
        width = self.heads * s.head_dim
        # Gemma 4 attention uses scale 1 on normalized q and k, so its logits are
        # large and sensitive to q/k rounding: norm, weight and RoPE run in fp32 and
        # round to BF16 once, before the head split (BF16 here raised decode KL by 30%).
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=HIFI2, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
        if len(rope) == 3:  # batch 1: ("compact", tensor, first tile) of the step's [cos | sin] (boundary.QkRope)
            roped = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
            self.qk_rope(xqkv, rope[1], self.rope_w, roped, cs_page0=rope[2])
            xqkv.deallocate(True)
            fused = roped
        else:
            rows = ttnn.reshape(xqkv, (1, 1, batch * self.heads, s.head_dim))
            normed = ttnn.rms_norm(rows, epsilon=self.eps, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=HIFI4)
            xqkv.deallocate(True)
            weighted = ttnn.mul(normed, self.head_weights(batch), memory_config=ttnn.L1_MEMORY_CONFIG)
            normed.deallocate(True)
            roped = apply_rope(weighted, *rope, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            weighted.deallocate(True)
            fused = ttnn.reshape(roped, (1, 1, batch, width))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=s.n_q, num_kv_heads=s.n_kv, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        roped.deallocate(True)

        ttnn.experimental.paged_update_cache(kv.k, k, update_idxs_tensor=cur_pos, page_table=page_table, **kv.update_args())
        ttnn.experimental.paged_update_cache(kv.v, v, update_idxs_tensor=cur_pos, page_table=page_table, **kv.update_args())
        k.deallocate(True)
        v.deallocate(True)

        grid = (8, 4) if s.head_dim >= 512 and batch == 1 else (8, 8)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv.k,
            kv.v,
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=1.0,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False
            ),
            compute_kernel_config=HIFI4,
            memory_config=DRAM,
            **({} if s.window is None else {"sliding_window_size": s.window}),
            **kv.sdpa_args(),
        )
        q.deallocate(True)
        return self._concat_heads_decode(attn, batch)  # [1, 1, 32, n_q * hd], one head per core

    def _concat_heads_decode(self, attn, batch):
        s = self.shape
        grid_x = min(batch, 8)
        while batch % grid_x:
            grid_x -= 1
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, batch // grid_x - 1))})
        shard = ttnn.create_sharded_memory_config(
            shape=(TILE, s.head_dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_sh = ttnn.to_memory_config(attn, shard)
        attn.deallocate(True)
        out = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=s.n_q)
        attn_sh.deallocate(True)
        return out
