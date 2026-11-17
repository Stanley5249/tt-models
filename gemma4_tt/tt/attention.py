"""Gemma 4 attention with heads split over the two N300 chips (TP=2).

Sliding layers: 16 Q heads, 8 KV heads, head_dim 256, window 1024.
Global layers: 16 Q heads, 2 KV heads, head_dim 512, K reused as V
(attention_k_eq_v), partial ("proportional") RoPE.

Each chip owns 8 Q heads and the KV heads they read, so attention needs no
communication until the output projection, whose partial sums are all-reduced.

KV caches are paged. Global layers address the vLLM page table. Sliding layers
use a per-slot ring of `window` tokens addressed by a slot page table, where
position p lives at ring index p % window. Decode writes and reads the ring
through `cache_position_modulo`; prefill builds the ring directly (see
`ring_selection`) so no write ever wraps inside one op.
"""

from dataclasses import dataclass

import torch
import ttnn

from gemma4_tt.tt.common import DRAM, HIFI2, HIFI4, TILE, WeightLoader, all_reduce, norm_weight


@dataclass(frozen=True)
class AttnShape:
    sliding: bool
    head_dim: int
    n_q: int  # per chip
    n_kv: int  # per chip
    window: int | None


def attn_shape(cfg, layer_idx: int, tp: int) -> AttnShape:
    sliding = cfg.layer_types[layer_idx] == "sliding_attention"
    if sliding:
        return AttnShape(True, cfg.head_dim, cfg.num_attention_heads // tp, cfg.num_key_value_heads // tp, cfg.sliding_window)
    return AttnShape(False, cfg.global_head_dim, cfg.num_attention_heads // tp, cfg.num_global_key_value_heads // tp, None)


def ring_selection(seq_len: int, padded_len: int, window: int) -> tuple[int, torch.Tensor] | None:
    """Selection matrix that turns the last tile-aligned K/V rows into a ring.

    Returns (tail_start, sel) with sel [1, 1, window, tail_len] so that
    ring = sel @ kv[:, :, tail_start:padded_len], ring[p % window] = kv[p]
    for the last `window` real positions p < seq_len. None when the prompt
    fits the window and the rows can be written as they are.
    """
    if seq_len <= window:
        return None
    tail_len = window + TILE
    tail_start = padded_len - tail_len
    sel = torch.zeros(window, tail_len)
    for p in range(seq_len - window, seq_len):
        sel[p % window, p - tail_start] = 1.0
    return tail_start, sel[None, None]


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
    return ttnn.matmul(x, _ROTATE[key], compute_kernel_config=HIFI4, memory_config=memory_config)


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config=DRAM) -> ttnn.Tensor:
    rot = rotate_half(x, memory_config)
    a = ttnn.mul(x, cos, memory_config=memory_config)
    out = ttnn.addcmul(a, rot, sin, memory_config=memory_config)  # x * cos + rot * sin
    rot.deallocate(True)
    a.deallocate(True)
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
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.q_norm = loader.put(f"L{layer_idx}.q_norm", lambda: norm_weight(ckpt.get(p + "q_norm.weight")), layout=rm)
        self.k_norm = loader.put(f"L{layer_idx}.k_norm", lambda: norm_weight(ckpt.get(p + "k_norm.weight")), layout=rm)
        # Decode normalizes all heads of the fused QKV row at once, so the head
        # weights become one [B * heads, hd] multiplier: q_norm rows, k_norm rows,
        # and ones for V (weightless norm). Built per batch size on first use.
        self._head_w = (ckpt.get(p + "q_norm.weight").float(), ckpt.get(p + "k_norm.weight").float())
        self._head_w_dev = {}
        self.mesh = loader.mesh

    # -- prefill -----------------------------------------------------------

    def prefill(self, x: ttnn.Tensor, rope, kv_cache, fill) -> ttnn.Tensor:
        """x [1, 1, S, H] replicated. Returns all-reduced attention output [1, 1, S, H].

        rope: (cos, sin) [1, 1, S, hd] for this layer type.
        fill: callable(k, v, kv_cache) that writes prefill K/V into the cache.
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
        fill(k, v, kv_cache)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=s.window,
            program_config=self._sdpa_prefill_config(q.shape[2]),
            compute_kernel_config=HIFI4,
            memory_config=DRAM,
        )
        for t in (q, k, v):
            t.deallocate(True)
        out = ttnn.experimental.nlp_concat_heads(attn, memory_config=DRAM)
        attn.deallocate(True)
        out2 = ttnn.linear(out, self.wo, compute_kernel_config=HIFI2, memory_config=DRAM)
        out.deallocate(True)
        return all_reduce(out2)

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
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=DRAM,
            )
        return self._head_w_dev[batch]

    def decode(self, x: ttnn.Tensor, rope, kv_cache, page_table, cur_pos) -> ttnn.Tensor:
        """x [1, 1, B, H] replicated. Returns all-reduced attention output [1, 1, B, H].

        rope: (cos, sin) [1, 1, B * heads, hd] in fused QKV row order, cos 1 and
        sin 0 on the V rows, so one rotation covers Q and K and leaves V alone.
        page_table: global layers get the vLLM table, sliding layers the ring table.
        cur_pos: int32 [B] absolute positions, -1 for inactive rows.
        """
        s = self.shape
        batch = x.shape[2]
        width = self.heads * s.head_dim
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=HIFI2, memory_config=ttnn.L1_MEMORY_CONFIG)
        rows = ttnn.reshape(xqkv, (1, 1, batch * self.heads, s.head_dim))
        normed = ttnn.rms_norm(rows, epsilon=self.eps, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=HIFI4)
        xqkv.deallocate(True)
        weighted = ttnn.mul(normed, self.head_weights(batch), memory_config=ttnn.L1_MEMORY_CONFIG)
        normed.deallocate(True)
        roped = apply_rope(weighted, *rope, memory_config=ttnn.L1_MEMORY_CONFIG)
        weighted.deallocate(True)
        fused = ttnn.reshape(roped, (1, 1, batch, width))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=s.n_q, num_kv_heads=s.n_kv, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        roped.deallocate(True)

        k_cache, v_cache = kv_cache
        modulo = {"cache_position_modulo": s.window} if s.sliding else {}
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos, page_table=page_table, **modulo)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos, page_table=page_table, **modulo)
        k.deallocate(True)
        v.deallocate(True)

        grid = (8, 4) if s.head_dim >= 512 and batch == 1 else (8, 8)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=s.window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False
            ),
            compute_kernel_config=HIFI4,
            memory_config=DRAM,
            **modulo,
        )
        q.deallocate(True)
        out = self._concat_heads_decode(attn, batch)
        out2 = ttnn.linear(out, self.wo, compute_kernel_config=HIFI2, memory_config=DRAM)
        out.deallocate(True)
        return all_reduce(out2)

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
        out_sh = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=s.n_q)
        attn_sh.deallocate(True)
        out = ttnn.sharded_to_interleaved(out_sh, DRAM)
        out_sh.deallocate(True)
        if out.shape[2] != batch:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, batch, out.shape[3]])
        return out
