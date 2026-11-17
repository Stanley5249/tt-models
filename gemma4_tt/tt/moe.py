"""Gemma 4 feed-forward block: shared dense MLP plus 128 routed experts (top-8).

Both the dense MLP (intermediate 2112) and every expert (intermediate 704)
are split over the intermediate dimension (TP=2), so each chip reads half of
every selected expert and the load is always balanced. The two partial sums
are reduced with one all-reduce on a [.., 2*H] concatenation, because each
branch has its own post-norm that needs the full sum.

Experts use indexed `sparse_matmul`. Decode (T <= 16) gives each (token,
slot) pair its own matmul slot. Prefill is expert-major: the host groups every
expert's tokens into chunks, so each chunk reads one expert's weights once.
"""

import torch
import ttnn

from gemma4_tt.tt.common import DRAM, HIFI2, HIFI4, TILE, WeightLoader, all_reduce, compute_config, norm_weight, rms_norm


# HF uses gelu_pytorch_tanh. Fusing GELU_TANH into the multiply halves the mean
# error of ttnn.gelu(fast_and_approximate_mode=True) and saves one op.
GELU_TANH = ttnn.UnaryOpType.GELU_TANH


def _matmul_1d_config(m_tiles: int, n: int, k: int, grid=(8, 8), in0_block_w: int | None = None):
    n_tiles = n // TILE
    # The 1D mcast kernel needs every core of a full rectangle to own work.
    for per_core_n in range(1, n_tiles + 1):
        cores = -(-n_tiles // per_core_n)
        shape = next(((x, cores // x) for x in range(grid[0], 0, -1) if cores % x == 0 and cores // x <= grid[1]), None)
        if shape:
            gx, gy = shape
            break
    # Large K blocks matter for these DRAM-bound decode shapes: measured in
    # scripts/bench_sparse_matmul.py, down (K=11 tiles) took 154 us at
    # in0_block_w=1 and 92 us at 11; gate_up (K=88) 194 us at 8 and 184 us at 22.
    k_tiles = k // TILE
    in0_block_w = in0_block_w or next(w for w in (22, 11, 8, 4, 2, 1) if k_tiles % w == 0)
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
    def __init__(
        self, loader: WeightLoader, ckpt, cfg, layer_idx: int, mlp_dtype=ttnn.bfloat8_b, expert_dtype=ttnn.bfloat4_b, expert_fidelity="LoFi"
    ):
        tp = loader.tp
        self.tp = tp
        self.expert_compute = compute_config(getattr(ttnn.MathFidelity, expert_fidelity))
        self.cfg = cfg
        self.hidden = H = cfg.hidden_size
        self.n_experts = E = cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.eps = cfg.rms_norm_eps
        self.mlp_local = cfg.intermediate_size // tp
        self.expert_local = cfg.moe_intermediate_size // tp
        # Each per-chip gate and up half is zero-padded to a multiple of 4 tiles
        # (352 -> 384 columns), so the gate_up sparse matmul has 24 output tiles and
        # runs on a full 8x3 grid: 172 us -> 127 us per decode layer, versus 8
        # cores for 22 tiles. The padding columns are sliced off before the down.
        self.expert_pad = -(-self.expert_local // (4 * TILE)) * 4 * TILE
        p = f"layers.{layer_idx}."
        put = loader.put
        rm = ttnn.ROW_MAJOR_LAYOUT

        def norm(name):
            return put(f"L{layer_idx}.{name}", lambda: norm_weight(ckpt.get(p + name + ".weight")), layout=rm)

        # The router norm, pre_ff and pre_ff2 all normalize the same residual, so
        # one weightless RMSNorm feeds three weight multiplies (TILE row vectors).
        def row(name):
            return put(f"L{layer_idx}.{name}.row", lambda: ckpt.get(p + name + ".weight").float().reshape(1, 1, 1, H))

        self.pre_ff = row("pre_feedforward_layernorm")
        self.pre_ff2 = row("pre_feedforward_layernorm_2")
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

        def experts_gate_up():
            gu = ckpt.get(p + "experts.gate_up_proj")  # [E, 2I, H], gate rows first
            inter = cfg.moe_intermediate_size
            g, u = gu[:, :inter].transpose(1, 2), gu[:, inter:].transpose(1, 2)  # [E, H, I]
            n, pad = self.expert_local, self.expert_pad - self.expert_local
            zeros = g.new_zeros(E, H, pad)
            parts = [torch.cat([g[..., c * n : (c + 1) * n], zeros, u[..., c * n : (c + 1) * n], zeros], -1) for c in range(tp)]
            return torch.cat(parts, -1)[None]

        def experts_down():
            return ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous()[None]  # [1, E, I, H]

        self.exp_gate_up = put(f"L{layer_idx}.exp_gate_up_p{self.expert_pad}", experts_gate_up, expert_dtype, shard_dim=-1)
        # per_expert_scale is applied to the routing weights: folding it into the BFP4
        # down weights changes their block rounding and measurably hurt accuracy.
        self.exp_down = put(f"L{layer_idx}.exp_down", experts_down, expert_dtype, shard_dim=-2)
        self.per_expert_scale = put(
            f"L{layer_idx}.per_expert_scale", lambda: ckpt.get(p + "router.per_expert_scale").float().reshape(1, 1, 1, E)
        )
        self.all_experts = ttnn.from_torch(
            torch.ones(1, 1, 1, E),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=loader.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(loader.mesh),
        )  # required operand in indexed mode, not read
        self.mesh = loader.mesh
        rep = ttnn.ReplicateTensorToMesh(loader.mesh)
        # Decode routing: topk_router_gpt does matmul -> top-k -> softmax over the k
        # logits, which equals Gemma's softmax -> top-k -> renormalize.
        self.router_2d = ttnn.reshape(self.router, (H, E))
        # per_expert_scale per slot comes from an embedding lookup of the expert id.
        self.scale_table = ttnn.from_torch(
            ckpt.get(p + "router.per_expert_scale").float()[:, None].expand(E, TILE).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=loader.mesh,
            mesh_mapper=rep,
        )
        self._zero_bias = {}
        self._combine_masks = {}
        self._chunk_cfgs = {}
        self.gate_up_cfg = _matmul_1d_config(1, 2 * self.expert_pad, H, in0_block_w=44)
        self.down_cfg = _matmul_1d_config(1, H, self.expert_local)

    def _rms(self, x, w):
        return rms_norm(x, w, self.eps)

    def topk_route(self, normed: ttnn.Tensor):
        """Top-k routing from the weightless RMSNorm of the residual.

        Returns renormalized weights [1, 1, T, k] and expert ids [1, 1, T, k] (uint16).
        """
        logits = ttnn.linear(normed, self.router, compute_kernel_config=HIFI4, memory_config=DRAM)
        probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI4)
        logits.deallocate(True)
        top_v, top_i = ttnn.topk(probs, k=self.top_k, dim=-1)
        scaled = ttnn.mul(probs, self.per_expert_scale)
        probs.deallocate(True)
        scaled_v = ttnn.gather(scaled, dim=-1, index=top_i)  # p[e] * per_expert_scale[e] for the chosen e
        scaled.deallocate(True)
        top_sum = ttnn.sum(top_v, dim=-1, keepdim=True)
        weights = ttnn.div(scaled_v, top_sum)
        for t in (top_v, scaled_v, top_sum):
            t.deallocate(True)
        return weights, top_i

    def _combine_mask(self, tokens: int) -> ttnn.Tensor:
        """[1, S, T, 1] with 1 where slot s belongs to token s // k."""
        if tokens not in self._combine_masks:
            slots = tokens * self.top_k
            mask = torch.zeros(1, slots, tokens, 1)
            for slot in range(slots):
                mask[0, slot, slot // self.top_k, 0] = 1.0
            self._combine_masks[tokens] = ttnn.from_torch(
                mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            )
        return self._combine_masks[tokens]

    def decode_constants(self, tokens: int):
        """Allocate the decode constants for `tokens` rows now (see Gemma4TT._decode_constants)."""
        self._bias(tokens)
        if tokens > 1:
            self._combine_mask(tokens)

    def _bias(self, tokens: int) -> ttnn.Tensor:
        if tokens not in self._zero_bias:
            self._zero_bias[tokens] = ttnn.from_torch(
                torch.zeros(tokens, self.n_experts),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        return self._zero_bias[tokens]

    def _experts_decode(self, x: ttnn.Tensor, normed: ttnn.Tensor) -> ttnn.Tensor:
        """Route and run the experts for T <= 32 tokens with compact (indexed) sparse matmuls.

        Slot s = 8 * token + j computes expert ids[token, j] for every token row.
        The slot weight (routing weight times per_expert_scale, and zero on rows
        of other tokens) scales the down input, so the combine is a plain sum.
        """
        tokens = x.shape[2]
        slots = tokens * self.top_k
        n = self.expert_local
        ids, weights = ttnn.experimental.topk_router_gpt(
            ttnn.reshape(normed, (tokens, self.hidden)),
            weight_tensor=self.router_2d,
            bias_tensor=self._bias(tokens),
            k=self.top_k,
            num_experts=self.n_experts,
        )  # row-major [T, k]: uint16 ids, bf16 softmax over the k logits
        if tokens == 1:
            idx = ttnn.reshape(ids, (1, slots))
            idx32 = ttnn.typecast(idx, ttnn.uint32)
        else:  # a uint16 reshape that moves rows is unsupported, so flatten in uint32
            idx32 = ttnn.reshape(ttnn.typecast(ids, ttnn.uint32), (1, slots))
            idx = ttnn.typecast(idx32, ttnn.uint16)
        scale = ttnn.embedding(idx32, self.scale_table, layout=ttnn.TILE_LAYOUT)  # [1, S, 32]
        w = ttnn.to_layout(ttnn.reshape(weights, (1, slots, 1)), ttnn.TILE_LAYOUT)  # [1, S, 1]
        w = ttnn.mul(scale, w)  # [1, S, 32], every column the slot weight
        w = ttnn.slice(ttnn.reshape(w, (1, slots, 1, TILE)), [0, 0, 0, 0], [1, slots, 1, 1])  # [1, S, 1, 1]
        if tokens > 1:
            w = ttnn.mul(w, self._combine_mask(tokens))  # [1, S, T, 1]
        gu = ttnn.sparse_matmul(
            x,
            self.exp_gate_up,
            sparsity=self.all_experts,
            indices=idx,
            program_config=self.gate_up_cfg,
            compute_kernel_config=self.expert_compute,
            memory_config=DRAM,
            dtype=ttnn.bfloat16,
        )  # [1, 1, 1, S, T, 2n]
        p = self.expert_pad
        gu = ttnn.reshape(gu, (1, slots, tokens, 2 * p))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, slots, tokens, n])
        up = ttnn.slice(gu, [0, 0, 0, p], [1, slots, tokens, p + n])
        gu.deallocate(True)
        h = ttnn.mul(gate, up, input_tensor_a_activations=[GELU_TANH])
        gate.deallocate(True)
        up.deallocate(True)
        hw = ttnn.mul(h, w)
        h.deallocate(True)
        w.deallocate(True)
        down = ttnn.sparse_matmul(
            hw,
            self.exp_down,
            sparsity=self.all_experts,
            indices=idx,
            program_config=self.down_cfg,
            compute_kernel_config=self.expert_compute,
            memory_config=DRAM,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            dtype=ttnn.bfloat16,
        )  # [1, S, T, H]
        hw.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1], compute_kernel_config=HIFI4, memory_config=DRAM)  # [1, 1, T, H]
        down.deallocate(True)
        return out

    def _mlp(self, x):
        gu = ttnn.linear(x, self.mlp_gate_up, compute_kernel_config=HIFI2, memory_config=DRAM)
        n = self.mlp_local
        t = gu.shape[2]
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, t, n])
        up = ttnn.slice(gu, [0, 0, 0, n], [1, 1, t, 2 * n])
        gu.deallocate(True)
        h = ttnn.mul(gate, up, input_tensor_a_activations=[GELU_TANH])
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(h, self.mlp_down, compute_kernel_config=HIFI2, memory_config=DRAM)
        h.deallocate(True)
        return out

    def _chunk_config(self, rows: int):
        """Sparse matmul configs for prefill chunks of `rows` token rows."""
        if rows not in self._chunk_cfgs:
            m = rows // TILE
            # Few K blocks: with packer L1 accumulation every block boundary rounds the
            # partial sum to the BF16 output (in0_block_w=8 raised prefill KL 0.32 -> 0.36).
            self._chunk_cfgs[rows] = (
                _matmul_1d_config(m, 2 * self.expert_pad, self.hidden, in0_block_w=44 if m <= 2 else 22),
                _matmul_1d_config(m, self.hidden, self.expert_local),
            )
        return self._chunk_cfgs[rows]

    def _plan_chunks(self, ids: torch.Tensor, weights: torch.Tensor, rows: int):
        """Host layout of expert-major prefill for top-k ids/weights [T, k].

        Each expert's (token, slot) pairs fill chunks of `rows` rows; padding rows
        gather token T (a zero row) and get weight 0. Returns the gather index into
        the token rows [C * rows], the row weights [C * rows], the expert of every
        chunk [C], and for each (slot j, token t) in slot-major order the row that
        holds its result [k * T].
        """
        tokens, k = ids.shape
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.n_experts)
        chunks = (counts + rows - 1) // rows
        chunk_expert = torch.repeat_interleave(torch.arange(self.n_experts), chunks)
        # first row of each expert's region, then the row of every sorted pair
        region = torch.cumsum(chunks * rows, 0) - chunks * rows
        sorted_e = flat[order]
        rank = torch.arange(flat.numel()) - (torch.cumsum(counts, 0) - counts)[sorted_e]
        row_of_sorted = region[sorted_e] + rank
        total = int(chunks.sum()) * rows
        gather = torch.full((total,), tokens, dtype=torch.int32)
        row_w = torch.zeros(total)
        gather[row_of_sorted] = (order // k).int()
        row_w[row_of_sorted] = weights.flatten()[order].float()
        row_of_pair = torch.empty(flat.numel(), dtype=torch.int32)
        row_of_pair[order] = row_of_sorted.int()
        back = row_of_pair.reshape(tokens, k).T.contiguous().flatten()  # slot-major
        return gather, row_w, chunk_expert, back

    def _experts_prefill(self, x: ttnn.Tensor, normed: ttnn.Tensor) -> ttnn.Tensor:
        """Expert-major experts for a prompt chunk x [1, 1, T, H].

        The host reads the top-k routing back, groups every expert's (token, slot)
        pairs into chunks of rows, and the device gathers those token rows, runs one
        indexed sparse matmul pair over the chunks (each expert's weights are read
        once per chunk, not once per 32-token group as with the union of experts),
        and gathers each token's k result rows back into a sum.
        """
        tokens = x.shape[2]
        k, n, p = self.top_k, self.expert_local, self.expert_pad
        weights, ids = self.topk_route(normed)
        w_host = ttnn.to_torch(ttnn.get_device_tensors(weights)[0]).reshape(-1, k)[:tokens]
        id_host = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1, k)[:tokens].long()
        weights.deallocate(True)
        ids.deallocate(True)
        per_expert = tokens * k / self.n_experts
        rows = 32 if per_expert <= 48 else 64 if per_expert <= 96 else 128
        gather, row_w, chunk_expert, back = self._plan_chunks(id_host, w_host, rows)
        n_chunks = chunk_expert.numel()

        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def up(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=rep, memory_config=DRAM)

        zero = ttnn.zeros([1, 1, TILE, self.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
        padded = ttnn.concat([x, zero], dim=2)  # row `tokens` is zero (x rows past it are never gathered)
        zero.deallocate(True)
        table = ttnn.to_layout(ttnn.reshape(padded, (tokens + TILE, self.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        padded.deallocate(True)
        g = up(gather[None], ttnn.uint32)
        xs = ttnn.embedding(g, table, layout=ttnn.TILE_LAYOUT)  # [1, C * rows, H]
        g.deallocate(True)
        table.deallocate(True)
        xs = ttnn.reshape(xs, (1, n_chunks, rows, self.hidden))
        w_dev = up(row_w.reshape(1, n_chunks, rows, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        gate_up_cfg, down_cfg = self._chunk_config(rows)

        outs = []
        for c0 in range(0, n_chunks, self.n_experts):  # indexed mode takes at most E slots per call
            c1 = min(c0 + self.n_experts, n_chunks)
            idx = up(chunk_expert[c0:c1][None].int(), ttnn.uint16)
            xc = xs if (c0, c1) == (0, n_chunks) else ttnn.slice(xs, [0, c0, 0, 0], [1, c1, rows, self.hidden])
            wc = w_dev if (c0, c1) == (0, n_chunks) else ttnn.slice(w_dev, [0, c0, 0, 0], [1, c1, rows, 1])
            gu = ttnn.sparse_matmul(
                xc,
                self.exp_gate_up,
                sparsity=self.all_experts,
                indices=idx,
                program_config=gate_up_cfg,
                compute_kernel_config=self.expert_compute,
                memory_config=DRAM,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                dtype=ttnn.bfloat16,
            )  # [1, c, rows, 2p]
            gate = ttnn.slice(gu, [0, 0, 0, 0], [1, c1 - c0, rows, n])
            up_ = ttnn.slice(gu, [0, 0, 0, p], [1, c1 - c0, rows, p + n])
            gu.deallocate(True)
            h = ttnn.mul(gate, up_, input_tensor_a_activations=[GELU_TANH])
            gate.deallocate(True)
            up_.deallocate(True)
            hw = ttnn.mul(h, wc)
            h.deallocate(True)
            outs.append(
                ttnn.sparse_matmul(
                    hw,
                    self.exp_down,
                    sparsity=self.all_experts,
                    indices=idx,
                    program_config=down_cfg,
                    compute_kernel_config=self.expert_compute,
                    memory_config=DRAM,
                    is_input_a_sparse=True,
                    is_input_b_sparse=True,
                    dtype=ttnn.bfloat16,
                )
            )  # [1, c, rows, H]
            hw.deallocate(True)
            idx.deallocate(True)
        xs.deallocate(True)
        w_dev.deallocate(True)
        down = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)
        if len(outs) > 1:
            for o in outs:
                o.deallocate(True)
        flat = ttnn.to_layout(ttnn.reshape(down, (n_chunks * rows, self.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        down.deallocate(True)
        b = up(back[None], ttnn.uint32)
        pairs = ttnn.embedding(b, flat, layout=ttnn.TILE_LAYOUT)  # [1, k * T, H], slot-major
        b.deallocate(True)
        flat.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.reshape(pairs, (1, k, tokens, self.hidden)), dims=[1], compute_kernel_config=HIFI4, memory_config=DRAM
        )
        pairs.deallocate(True)
        return out

    def _experts(self, x, normed, chunk_tokens: int = 4096):
        """Prefill experts in token chunks, which bound the gathered-row buffers."""
        t = x.shape[2]
        if t <= chunk_tokens:
            return self._experts_prefill(x, normed)
        outs = []
        for start in range(0, t, chunk_tokens):
            end = min(start + chunk_tokens, t)
            xs = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, self.hidden])
            ns = ttnn.slice(normed, [0, 0, start, 0], [1, 1, end, self.hidden])
            outs.append(self._experts_prefill(xs, ns))
            xs.deallocate(True)
            ns.deallocate(True)
        out = ttnn.concat(outs, dim=2)
        for o in outs:
            o.deallocate(True)
        return out

    def __call__(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """residual [1, 1, T, H] replicated. Returns the block output to add to the residual."""
        t = residual.shape[2]
        normed = rms_norm(residual, None, self.eps)
        mlp_in = ttnn.mul(normed, self.pre_ff)
        mlp = self._mlp(mlp_in)
        mlp_in.deallocate(True)

        exp_in = ttnn.mul(normed, self.pre_ff2)
        # Indexed mode allows at most one slot per expert (T * k <= E, so T <= 16);
        # longer inputs, such as the 32-token prefill warmup, take the grouped path.
        if t * self.top_k <= self.n_experts:
            experts = self._experts_decode(exp_in, normed)
        else:
            experts = self._experts(exp_in, normed)
        exp_in.deallocate(True)
        normed.deallocate(True)
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
