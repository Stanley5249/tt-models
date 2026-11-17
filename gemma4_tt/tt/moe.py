"""Gemma 4 feed-forward block: shared dense MLP plus 128 routed experts (top-8).

Both the dense MLP (intermediate 2112) and every expert (intermediate 704)
are split over the intermediate dimension (TP=2), so each chip reads half of
every selected expert and the load is always balanced. The two partial sums
are reduced with one all-reduce on a [.., 2*H] concatenation, because each
branch has its own post-norm that needs the full sum.

Experts use `sparse_matmul`. Tokens are packed in groups of 32 rows; a group
computes the union of the experts its tokens selected, and routing weights of
zero cancel the unselected (token, expert) pairs, so the result is exact.
"""

import torch
import ttnn

from gemma4_tt.tt.common import DRAM, HIFI2, HIFI4, LOFI, TILE, WeightLoader, all_reduce, norm_weight


def _matmul_1d_config(m_tiles: int, n: int, k: int, grid=(8, 8)):
    n_tiles = n // TILE
    # The 1D mcast kernel needs every core of a full rectangle to own work.
    for per_core_n in range(1, n_tiles + 1):
        cores = -(-n_tiles // per_core_n)
        shape = next(((x, cores // x) for x in range(grid[0], 0, -1) if cores % x == 0 and cores // x <= grid[1]), None)
        if shape:
            gx, gy = shape
            break
    in0_block_w = 1
    for w in (8, 4, 2, 1):
        if (k // TILE) % w == 0:
            in0_block_w = w
            break
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


class FeedForward:
    def __init__(self, loader: WeightLoader, ckpt, cfg, layer_idx: int, mlp_dtype=ttnn.bfloat8_b, expert_dtype=ttnn.bfloat4_b):
        tp = loader.tp
        self.tp = tp
        self.cfg = cfg
        self.hidden = H = cfg.hidden_size
        self.n_experts = E = cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.eps = cfg.rms_norm_eps
        self.mlp_local = cfg.intermediate_size // tp
        self.expert_local = cfg.moe_intermediate_size // tp
        p = f"layers.{layer_idx}."
        put = loader.put
        rm = ttnn.ROW_MAJOR_LAYOUT

        def norm(name):
            return put(f"L{layer_idx}.{name}", lambda: norm_weight(ckpt.get(p + name + ".weight")), layout=rm)

        self.pre_ff = norm("pre_feedforward_layernorm")
        self.pre_ff2 = norm("pre_feedforward_layernorm_2")
        self.post_ff1 = norm("post_feedforward_layernorm_1")
        self.post_ff2 = norm("post_feedforward_layernorm_2")
        self.post_ff = norm("post_feedforward_layernorm")

        def mlp_gate_up():
            g = ckpt.get(p + "mlp.gate_proj.weight").T  # [H, I]
            u = ckpt.get(p + "mlp.up_proj.weight").T
            n = self.mlp_local
            return torch.cat([torch.cat([g[:, c * n : (c + 1) * n], u[:, c * n : (c + 1) * n]], -1) for c in range(tp)], -1)[None, None]

        self.mlp_gate_up = put(f"L{layer_idx}.mlp_gate_up", mlp_gate_up, mlp_dtype, shard_dim=-1)
        self.mlp_down = put(
            f"L{layer_idx}.mlp_down", lambda: ckpt.get(p + "mlp.down_proj.weight").T.contiguous()[None, None], mlp_dtype, shard_dim=-2
        )

        def router():
            # HF: softmax((rmsnorm(x) * scale * H^-0.5) @ proj.T); fold scale and H^-0.5 into proj.
            w = ckpt.get(p + "router.proj.weight").float()  # [E, H]
            scale = ckpt.get(p + "router.scale").float() * H**-0.5
            return (w * scale[None, :]).T.contiguous()[None, None]

        self.router = put(f"L{layer_idx}.router", router, ttnn.bfloat16)
        self.per_expert_scale = put(
            f"L{layer_idx}.per_expert_scale", lambda: ckpt.get(p + "router.per_expert_scale").float().reshape(1, 1, 1, E)
        )

        def experts_gate_up():
            gu = ckpt.get(p + "experts.gate_up_proj")  # [E, 2I, H], gate rows first
            inter = cfg.moe_intermediate_size
            g, u = gu[:, :inter].transpose(1, 2), gu[:, inter:].transpose(1, 2)  # [E, H, I]
            n = self.expert_local
            return torch.cat([torch.cat([g[..., c * n : (c + 1) * n], u[..., c * n : (c + 1) * n]], -1) for c in range(tp)], -1)[None]

        def experts_down():
            return ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous()[None]  # [1, E, I, H]

        self.exp_gate_up = put(f"L{layer_idx}.exp_gate_up", experts_gate_up, expert_dtype, shard_dim=-1)
        self.exp_down = put(f"L{layer_idx}.exp_down", experts_down, expert_dtype, shard_dim=-2)
        self.gate_up_cfg = _matmul_1d_config(1, 2 * self.expert_local, H)
        self.down_cfg = _matmul_1d_config(1, H, self.expert_local)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.eps, memory_config=DRAM, compute_kernel_config=HIFI4)

    def route(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Dense routing weights [1, 1, T, E]: top-k renormalized probabilities, zeros elsewhere."""
        normed = ttnn.rms_norm(residual, epsilon=self.eps, memory_config=DRAM, compute_kernel_config=HIFI4)
        logits = ttnn.linear(normed, self.router, compute_kernel_config=HIFI4, memory_config=DRAM)
        normed.deallocate(True)
        probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI4)
        logits.deallocate(True)
        top_v, top_i = ttnn.topk(probs, k=self.top_k, dim=-1)
        top_sum = ttnn.sum(top_v, dim=-1, keepdim=True)
        top_v2 = ttnn.div(top_v, top_sum)
        dense = ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=top_i, src=top_v2)
        for t in (probs, top_v, top_i, top_sum, top_v2):
            t.deallocate(True)
        out = ttnn.mul(dense, self.per_expert_scale)
        dense.deallocate(True)
        return out

    def _mlp(self, x):
        gu = ttnn.linear(x, self.mlp_gate_up, compute_kernel_config=HIFI2, memory_config=DRAM)
        n = self.mlp_local
        t = gu.shape[2]
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, t, n])
        up = ttnn.slice(gu, [0, 0, 0, n], [1, 1, t, 2 * n])
        gu.deallocate(True)
        act = ttnn.gelu(gate, fast_and_approximate_mode=True)
        h = ttnn.mul(act, up)
        for x_ in (gate, up, act):
            x_.deallocate(True)
        out = ttnn.linear(h, self.mlp_down, compute_kernel_config=HIFI2, memory_config=DRAM)
        h.deallocate(True)
        return out

    def _experts_group(self, x, routing, groups):
        """x [1, G, M, H], routing [1, G, M, E] with M <= 32. Returns partial expert output [G, 1, M, H]."""
        E, n = self.n_experts, self.expert_local
        m = x.shape[2]
        hit = ttnn.sum(routing, dim=2, keepdim=True)  # [1, G, 1, E], > 0 where any token uses the expert
        hit = ttnn.gtz(hit)
        sparsity = ttnn.to_layout(hit, ttnn.ROW_MAJOR_LAYOUT)
        hit.deallocate(True)
        gu = ttnn.sparse_matmul(
            x,
            self.exp_gate_up,
            sparsity=sparsity,
            program_config=self.gate_up_cfg,
            compute_kernel_config=LOFI,
            memory_config=DRAM,
            dtype=ttnn.bfloat16,
        )  # [1, G, 1, E, M, 2n]
        gu = ttnn.reshape(gu, (groups, E, m, 2 * n))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [groups, E, m, n])
        up = ttnn.slice(gu, [0, 0, 0, n], [groups, E, m, 2 * n])
        gu.deallocate(True)
        act = ttnn.gelu(gate, fast_and_approximate_mode=True)
        h = ttnn.mul(act, up)
        for t in (gate, up, act):
            t.deallocate(True)
        sparsity2 = ttnn.reshape(sparsity, (1, 1, groups, E))
        down = ttnn.sparse_matmul(
            h,
            self.exp_down,
            sparsity=sparsity2,
            program_config=self.down_cfg,
            compute_kernel_config=LOFI,
            memory_config=DRAM,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            dtype=ttnn.bfloat16,
        )  # [G, E, M, H]
        h.deallocate(True)
        sparsity.deallocate(True)
        w = ttnn.permute(routing, (1, 3, 2, 0))  # [G, E, M, 1]
        weighted = ttnn.mul(down, w)
        down.deallocate(True)
        w.deallocate(True)
        out = ttnn.sum(weighted, dim=1, keepdim=True)  # [G, 1, M, H]
        weighted.deallocate(True)
        return out

    def _experts(self, x, routing, chunk_tokens: int):
        t = x.shape[2]
        if t <= TILE:
            return self._experts_group(x, routing, 1)
        outs = []
        for start in range(0, t, chunk_tokens):
            end = min(start + chunk_tokens, t)
            g = (end - start) // TILE
            xs = ttnn.reshape(ttnn.slice(x, [0, 0, start, 0], [1, 1, end, self.hidden]), (1, g, TILE, self.hidden))
            rs = ttnn.reshape(ttnn.slice(routing, [0, 0, start, 0], [1, 1, end, self.n_experts]), (1, g, TILE, self.n_experts))
            o = self._experts_group(xs, rs, g)
            outs.append(ttnn.reshape(o, (1, 1, end - start, self.hidden)))
        out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        return out

    def __call__(self, residual: ttnn.Tensor, chunk_tokens: int = 128) -> ttnn.Tensor:
        """residual [1, 1, T, H] replicated. Returns the block output to add to the residual."""
        t = residual.shape[2]
        mlp_in = self._rms(residual, self.pre_ff)
        mlp = self._mlp(mlp_in)
        mlp_in.deallocate(True)

        routing = self.route(residual)
        exp_in = self._rms(residual, self.pre_ff2)
        experts = self._experts(exp_in, routing, chunk_tokens)
        exp_in.deallocate(True)
        routing.deallocate(True)
        if experts.shape[2] != t:
            experts = ttnn.slice(experts, [0, 0, 0, 0], [1, 1, t, self.hidden])

        both = ttnn.concat([mlp, experts], dim=-1)
        mlp.deallocate(True)
        experts.deallocate(True)
        both = all_reduce(both)
        mlp = ttnn.slice(both, [0, 0, 0, 0], [1, 1, t, self.hidden])
        experts = ttnn.slice(both, [0, 0, 0, self.hidden], [1, 1, t, 2 * self.hidden])
        both.deallocate(True)
        a = self._rms(mlp, self.post_ff1)
        b = self._rms(experts, self.post_ff2)
        mlp.deallocate(True)
        experts.deallocate(True)
        s = ttnn.add(a, b)
        a.deallocate(True)
        b.deallocate(True)
        out = self._rms(s, self.post_ff)
        s.deallocate(True)
        return out
