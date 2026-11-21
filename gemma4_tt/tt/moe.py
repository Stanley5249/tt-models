"""Gemma 4 feed-forward block: shared dense MLP plus 128 routed experts (top-8).

Both the dense MLP (intermediate 2112) and every expert (intermediate 704)
are split over the intermediate dimension (TP=2), so each chip reads half of
every selected expert and the load is always balanced. The two partial sums
are reduced with one all-reduce on a [.., 2*H] concatenation, because each
branch has its own post-norm that needs the full sum.

Decode streams the routed experts of all its tokens (T <= 4) per DRAM bank
with two generic ops (expert_stream.py). Prefill is expert-major: the host
groups every expert's tokens into chunks, so each chunk reads one expert's
weights once. The down weight is stored with the hidden padded to 3072 columns
for the stream's bank layout; prefill slices the padding off.
"""

import torch
import ttnn

from gemma4_tt.tt.common import DRAM, HIFI2, HIFI4, L1, TILE, WeightLoader, all_reduce, compute_config, norm_weight, rms_norm
from gemma4_tt.tt.boundary import Router
from gemma4_tt.tt.expert_q4 import Q4Experts
from gemma4_tt.tt.expert_stream import ExpertStream, MlpStream, pad_down, run

_STREAMS: dict[tuple, object] = {}


# Decode batches up to this many tokens take the expert streams (their scratch holds 32 slots).
STREAM_ROWS = 4


def expert_stream(mesh, cfg, expert_local: int, rows: int = 1) -> ExpertStream:
    """One ExpertStream (core placement) per mesh and token count, shared by the layers."""
    key = ("experts", id(mesh), rows)
    if key not in _STREAMS:
        _STREAMS[key] = ExpertStream(mesh, cfg.hidden_size, expert_local, cfg.top_k_experts, cfg.num_experts, rows=rows)
    return _STREAMS[key]


def router_op(mesh, cfg, rows: int = 1) -> Router:
    key = ("router", id(mesh), rows)
    if key not in _STREAMS:
        _STREAMS[key] = Router(mesh, cfg.hidden_size, cfg.num_experts, cfg.top_k_experts, rows=rows)
    return _STREAMS[key]


def mlp_stream(mesh, cfg, mlp_pad: int) -> MlpStream:
    key = ("mlp", id(mesh))
    if key not in _STREAMS:
        # leaves the router's cores free: router and MLP gate|up run as one merged op
        _STREAMS[key] = MlpStream(mesh, cfg.hidden_size, mlp_pad, exclude=router_op(mesh, cfg).cores)
    return _STREAMS[key]


# Expert-major prefill rounds its chunk count up to a multiple of this (see _plan_chunks).
CHUNK_GRANULE = 16

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
        self, loader: WeightLoader, ckpt, cfg, layer_idx: int, mlp_dtype=ttnn.bfloat8_b, expert_dtype=ttnn.bfloat4_b, expert_fidelity="LoFi",
        expert_q4: bool = False,
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
        # The per-chip MLP gate and up halves are zero-padded 1056 -> 1152 columns
        # (and the down weight's K with zero rows), so the down matmul's K is 36
        # tiles: 44 -> 31 us per layer with the default program config.
        self.mlp_pad = -(-self.mlp_local // (4 * TILE)) * 4 * TILE
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

        self.mlp_stream = mlp_stream(loader.mesh, cfg, self.mlp_pad)

        def mlp_gate_up():
            g = ckpt.get(p + "mlp.gate_proj.weight").T  # [H, I]
            u = ckpt.get(p + "mlp.up_proj.weight").T
            n, zeros = self.mlp_local, g.new_zeros(H, self.mlp_pad - self.mlp_local)
            parts = [torch.cat([g[:, c * n : (c + 1) * n], zeros, u[:, c * n : (c + 1) * n], zeros], -1) for c in range(tp)]
            return torch.cat(parts, -1)[None, None]

        def mlp_down():
            d = ckpt.get(p + "mlp.down_proj.weight").T  # [I, H]
            n, zeros = self.mlp_local, d.new_zeros(self.mlp_pad - self.mlp_local, H)
            d = torch.cat([torch.cat([d[c * n : (c + 1) * n], zeros]) for c in range(tp)])
            return pad_down(d, self.mlp_stream.down_width).contiguous()[None, None]  # hidden padded for the stream

        self.mlp_gate_up = put(f"L{layer_idx}.mlp_gate_up_p{self.mlp_pad}", mlp_gate_up, mlp_dtype, shard_dim=-1)
        self.mlp_down = put(f"L{layer_idx}.mlp_down_p{self.mlp_pad}_w{self.mlp_stream.down_width}", mlp_down, mlp_dtype, shard_dim=-2)

        def router():
            # HF: softmax((rmsnorm(x) * scale * H^-0.5) @ proj.T); fold scale and H^-0.5 into proj.
            w = ckpt.get(p + "router.proj.weight").float()  # [E, H]
            scale = ckpt.get(p + "router.scale").float() * H**-0.5
            return (w * scale[None, :]).T.contiguous()[None, None]

        self.router = put(f"L{layer_idx}.router", router, ttnn.bfloat16)

        def assemble_gate_up(t):
            """[E, H, 2I] (gate columns, then up) -> the device order: per chip, gate c, padding, up c, padding."""
            inter = cfg.moe_intermediate_size
            g, u = t[..., :inter], t[..., inter:]
            n, pad = self.expert_local, self.expert_pad - self.expert_local
            zeros = g.new_zeros(*g.shape[:-1], pad)
            return torch.cat([torch.cat([g[..., c * n : (c + 1) * n], zeros, u[..., c * n : (c + 1) * n], zeros], -1) for c in range(tp)], -1)

        def experts_gate_up():
            return assemble_gate_up(ckpt.get(p + "experts.gate_up_proj").transpose(1, 2))[None]  # [E, 2I, H] -> [1, E, H, 2I]

        self.stream = expert_stream(loader.mesh, cfg, self.expert_local)
        self.pack_cores = [ttnn.CoreCoord(x, 7) for x in range(8)]
        self.down_width = self.stream.down_width

        def experts_down():
            down = ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous()  # [E, I, H]
            return pad_down(down, self.down_width)[None]  # [1, E, I, 3072]

        # Experimental (expert_q4.py): the checkpoint's Q4_0 integers and steps, for batch-1 decode.
        # The prefill sparse matmuls cannot apply the steps and two copies of the experts do not fit,
        # so prefill then runs the batch-1 streams token by token (slow; for accuracy runs only).
        self.q4 = Q4Experts(loader, ckpt, layer_idx, self.stream, assemble_gate_up, self.expert_pad) if expert_q4 else None
        if self.q4 is None:
            self.exp_gate_up = put(f"L{layer_idx}.exp_gate_up_p{self.expert_pad}", experts_gate_up, expert_dtype, shard_dim=-1)
            # per_expert_scale is applied to the routing weights: folding it into the BFP4
            # down weights changes their block rounding and measurably hurt accuracy.
            self.exp_down = put(f"L{layer_idx}.exp_down_p{self.down_width}", experts_down, expert_dtype, shard_dim=-2)
        # Row-major per_expert_scale for the stream's gate_up writer.
        self.scale_row = ttnn.from_torch(
            ckpt.get(p + "router.per_expert_scale").float().reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=loader.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(loader.mesh),
        )
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
        # Decode routing: topk_router_gpt does matmul -> top-k -> softmax over the k
        # logits, which equals Gemma's softmax -> top-k -> renormalize.
        self.router_2d = ttnn.reshape(self.router, (H, E))
        self._zero_bias = {}
        self._chunk_cfgs = {}

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

    def decode_constants(self, tokens: int):
        """Allocate the decode constants for `tokens` rows now (see Gemma4TT._prepare_traces)."""
        self._bias(tokens)

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

    def _route(self, normed: ttnn.Tensor, tokens: int):
        """topk_router_gpt: row-major [T, k] uint16 ids and BF16 softmax weights over the k logits."""
        return ttnn.experimental.topk_router_gpt(
            ttnn.reshape(normed, (tokens, self.hidden)),
            weight_tensor=self.router_2d,
            bias_tensor=self._bias(tokens),
            k=self.top_k,
            num_experts=self.n_experts,
        )

    def _experts_q4_rows(self, x: ttnn.Tensor, normed: ttnn.Tensor, group: int = 256) -> ttnn.Tensor:
        """Q4_0 experts for every row of x [1, 1, T, H], one batch-1 stream pair per row (slow).

        Rows are concatenated in groups: one ttnn.concat takes at most about 2,000 inputs (two
        runtime arguments each), and the one-row tensors are freed group by group."""
        groups = []
        for g0 in range(0, x.shape[2], group):
            outs = []
            for t in range(g0, min(g0 + group, x.shape[2])):
                xr = ttnn.slice(x, [0, 0, t, 0], [1, 1, t + 1, self.hidden], memory_config=L1)
                nr = ttnn.slice(normed, [0, 0, t, 0], [1, 1, t + 1, self.hidden], memory_config=L1)
                ids, weights = self._route(nr, 1)
                act = self.stream.act_tensor()
                self.q4.gate_up_op(xr, ids, weights, self.scale_row, act)
                out = self.stream.out_tensor(DRAM)
                self.q4.down_op(act, ids, out)
                for u in (xr, nr, ids, weights, act):
                    u.deallocate(True)
                outs.append(out)
            groups.append(self._concat_rows(outs))
        return self._concat_rows(groups)

    @staticmethod
    def _concat_rows(parts: list[ttnn.Tensor]) -> ttnn.Tensor:
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=2)
        for t in parts:
            t.deallocate(True)
        return out

    def _unpad(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """Drop the down weight's padding columns."""
        out = ttnn.slice(t, [0, 0, 0, 0], [1, 1, t.shape[2], self.hidden])
        t.deallocate(True)
        return out

    def _mlp(self, x, dtype=None):
        gu = ttnn.linear(x, self.mlp_gate_up, compute_kernel_config=HIFI2, memory_config=DRAM)
        n = self.mlp_pad  # padding columns are zero in gate and up, so h carries zeros there
        t = gu.shape[2]
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, t, n])
        up = ttnn.slice(gu, [0, 0, 0, n], [1, 1, t, 2 * n])
        gu.deallocate(True)
        h = ttnn.mul(gate, up, input_tensor_a_activations=[GELU_TANH])
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(h, self.mlp_down, compute_kernel_config=HIFI2, memory_config=DRAM, dtype=dtype)
        h.deallocate(True)
        return self._unpad(out)

    def decode_compact(self, normed, mlp_in, exp_in, boundary) -> ttnn.Tensor:
        """1-4 decode tokens on compact boundaries (boundary.py). normed, mlp_in, exp_in: row form BF16
        [1, 1, rows, H] from the attention tail. Returns this chip's [mlp rows | expert rows] partial sums,
        compact FP32 (2 * rows vectors)."""
        rows = normed.shape[2]
        out = boundary.compact(2 * rows)
        ms = self.mlp_stream
        h = ms.h_tensor(rows)
        router = router_op(self.mesh, self.cfg, rows)
        ids, weights = router.outputs()
        # The router is latency-bound and the MLP gate|up DRAM-bound: one merged op on disjoint cores.
        run([router(normed, self.router, ids, weights, build=True), ms.gate_up(mlp_in, self.mlp_gate_up, h, rows=rows, build=True)])
        ms.down(h, self.mlp_down, out, compact_chunk0=0, rows=rows)
        h.deallocate(True)
        # At 2-4 rows the router op replaced topk_router_gpt and the reshape and typecast ops that
        # flattened its [T, k] output: a batch-4 step went from 34.8 to 33.8 ms.
        st = self.stream if rows == 1 else expert_stream(self.mesh, self.cfg, self.expert_local, rows)
        act = st.act_tensor()
        expert_chunk0 = rows * boundary.V * 64
        if self.q4 is not None:
            assert rows == 1, "the Q4_0 experts are batch-1 only"
            self.q4.gate_up_op(exp_in, ids, weights, self.scale_row, act)
            weights.deallocate(True)
            self.q4.down_op(act, ids, out, compact_chunk0=expert_chunk0)
        else:
            st.gate_up(exp_in, ids, weights, self.scale_row, self.exp_gate_up, act)
            weights.deallocate(True)
            st.down(act, ids, self.exp_down, out, compact_chunk0=expert_chunk0)
        act.deallocate(True)
        ids.deallocate(True)
        return out

    def _chunk_config(self, rows: int):
        """Sparse matmul configs for prefill chunks of `rows` token rows."""
        if rows not in self._chunk_cfgs:
            m = rows // TILE
            # Few K blocks: with packer L1 accumulation every block boundary rounds the
            # partial sum to the BF16 output (in0_block_w=8 raised prefill KL 0.32 -> 0.36).
            self._chunk_cfgs[rows] = (
                _matmul_1d_config(m, 2 * self.expert_pad, self.hidden, in0_block_w=44 if m <= 2 else 22),
                _matmul_1d_config(m, self.down_width, self.expert_local),
            )
        return self._chunk_cfgs[rows]

    def chunk_rows(self, tokens: int) -> int:
        """Rows per expert chunk for a prefill piece of `tokens` tokens.

        One tile of rows wastes the least on padding, and the sparse matmuls' time
        follows their rows: a 2,048-token piece takes 1.28 s with 32-row chunks,
        1.34 s with 64 and 1.54 s with 128 (README text, see docs/serving.md)."""
        return TILE

    def _plan_chunks(self, ids: torch.Tensor, weights: torch.Tensor, rows: int):
        """Host layout of expert-major prefill for top-k ids/weights [T, k].

        Each expert's (token, slot) pairs fill chunks of `rows` rows; padding rows
        gather token 0 and get weight 0, which zeroes their results. The chunk count rounds up to a
        multiple of CHUNK_GRANULE with zero chunks: it depends on the routing, and
        every new count would compile new programs. Returns the gather index into
        the token rows [C * rows], the row weights [C * rows], the expert of every
        chunk [C], and for each (slot j, token t) in slot-major order the row that
        holds its result [k * T].
        """
        tokens, k = ids.shape
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.n_experts)
        chunks = (counts + rows - 1) // rows
        real = int(chunks.sum())
        extra = -(-real // CHUNK_GRANULE) * CHUNK_GRANULE - real
        chunk_expert = torch.cat([torch.repeat_interleave(torch.arange(self.n_experts), chunks), torch.zeros(extra, dtype=torch.long)])
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
        rows = self.chunk_rows(tokens)
        gather, row_w, chunk_expert, back = self._plan_chunks(id_host, w_host, rows)
        n_chunks = chunk_expert.numel()

        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def up(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=rep, memory_config=DRAM)

        table = ttnn.to_layout(ttnn.reshape(x, (tokens, self.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        gate_up_cfg, down_cfg = self._chunk_config(rows)

        outs = []
        for c0 in range(0, n_chunks, self.n_experts):  # indexed mode takes at most E slots per call
            c1 = min(c0 + self.n_experts, n_chunks)
            # Gathering each call's rows separately saves slicing one big gather.
            g = up(gather[c0 * rows : c1 * rows][None], ttnn.uint32)
            xc = ttnn.reshape(ttnn.embedding(g, table, layout=ttnn.TILE_LAYOUT), (1, c1 - c0, rows, self.hidden))
            g.deallocate(True)
            wc = up(row_w[c0 * rows : c1 * rows].reshape(1, c1 - c0, rows, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            idx = up(chunk_expert[c0:c1][None].int(), ttnn.uint16)
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
            xc.deallocate(True)
            wc.deallocate(True)
        table.deallocate(True)
        down = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)
        if len(outs) > 1:
            for o in outs:
                o.deallocate(True)
        flat = ttnn.to_layout(ttnn.reshape(down, (n_chunks * rows, self.down_width)), ttnn.ROW_MAJOR_LAYOUT)
        down.deallocate(True)
        b = up(back[None], ttnn.uint32)
        pairs = ttnn.embedding(b, flat, layout=ttnn.TILE_LAYOUT)  # [1, k * T, 3072], slot-major
        b.deallocate(True)
        flat.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.reshape(pairs, (1, k, tokens, self.down_width)), dims=[1], compute_kernel_config=HIFI4, memory_config=DRAM
        )
        pairs.deallocate(True)
        return self._unpad(out)

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
        """Prefill: residual [1, 1, T, H] replicated. Returns the block output to add to the residual.
        (Decode runs decode_compact.)"""
        t = residual.shape[2]
        normed = rms_norm(residual, None, self.eps)
        mlp_in = ttnn.mul(normed, self.pre_ff, memory_config=DRAM)
        mlp = self._mlp(mlp_in)
        mlp_in.deallocate(True)

        exp_in = ttnn.mul(normed, self.pre_ff2, memory_config=DRAM)
        experts = self._experts_q4_rows(exp_in, normed) if self.q4 is not None else self._experts(exp_in, normed)
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
