"""Gemma 4 26B-A4B text model on a 1x2 Wormhole mesh.

The token embedding stays on the host (1.5 GB BF16 that would otherwise take
device memory); the tied LM head is uploaded once, split over the vocabulary.

The model holds no per-request state: every K/V row lives in the paged cache
(kv_cache.py) at the blocks the caller's page tables name, so requests can be
moved, preempted, resumed and share cached prefixes without telling the model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import ttnn
from transformers.models.gemma4 import modeling_gemma4 as hf

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.reference import embed_scale
from gemma4_tt.tt.attention import CHUNK_ALIGN, Attention, PrefillChunk
from gemma4_tt.tt.boundary import Boundary, compact_tensor, to_compact
from gemma4_tt.tt.kv_cache import KvCache
from gemma4_tt.tt.common import (
    DRAM,
    HIFI2,
    HIFI4,
    TILE,
    WeightLoader,
    all_gather_compact,
    dram_sharded,
    init_ccl,
    norm_weight,
    rms_norm,
    width_sharded_l1,
)

# The LM head is read as LM_SPLITS DRAM-sharded matmuls per chip: 2.5 -> 1.37 ms per
# decode step against one interleaved matmul (scripts/bench_lm_head_dram.py).
# A single DRAM-sharded matmul over all 65536 columns does not fit L1.
LM_SPLITS = 16
LM_CORES = 8
# Longest piece of a prompt one prefill pass computes. It bounds activation memory
# (the expert-major MoE gathers k = 8 rows per token) whatever chunk vLLM schedules.
PREFILL_CHUNK = 4096

from gemma4_tt.tt.moe import STREAM_ROWS, FeedForward  # noqa: E402


@dataclass
class Precision:
    attention: ttnn.DataType = ttnn.bfloat8_b
    mlp: ttnn.DataType = ttnn.bfloat8_b
    experts: ttnn.DataType = ttnn.bfloat4_b
    lm_head: ttnn.DataType = ttnn.bfloat8_b
    kv_cache: ttnn.DataType = ttnn.bfloat8_b
    expert_fidelity: str = "LoFi"  # compute fidelity of the expert sparse matmuls

    @classmethod
    def parse(cls, spec: str) -> "Precision":
        """From "experts=bfp4,expert_fidelity=HiFi2"-style overrides."""
        names = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        kwargs = {}
        for item in filter(None, spec.split(",")):
            key, value = item.split("=")
            kwargs[key] = value if key.endswith("fidelity") else names[value]
        return cls(**kwargs)

    def tag(self) -> str:
        names = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}
        return "-".join(names[d] for d in (self.attention, self.mlp, self.experts, self.lm_head))


class DecoderLayer:
    def __init__(self, loader, ckpt, cfg, idx, precision: Precision):
        p = f"layers.{idx}."
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.idx = idx
        self.sliding = cfg.layer_types[idx] == "sliding_attention"
        self.eps = cfg.rms_norm_eps
        self.attn = Attention(loader, ckpt, cfg, idx, precision.attention)
        self.ff = FeedForward(loader, ckpt, cfg, idx, precision.mlp, precision.experts, precision.expert_fidelity)
        self.input_ln = loader.put(f"L{idx}.input_ln", lambda: norm_weight(ckpt.get(p + "input_layernorm.weight")), layout=rm)
        self.post_attn_ln = loader.put(
            f"L{idx}.post_attn_ln", lambda: norm_weight(ckpt.get(p + "post_attention_layernorm.weight")), layout=rm
        )
        self.scalar = float(ckpt.get(p + "layer_scalar").float().item())
        self._raw = {n: ckpt.get(p + n + ".weight").float() for n in (
            "post_attention_layernorm", "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "post_feedforward_layernorm", "input_layernorm",
        )}

    def boundary_weights(self, mesh, next_norm: torch.Tensor):
        """Compact FP32 weights of the two batch-1 tails; next_norm: the next layer's input norm (or the final norm)."""
        r = self._raw
        attn = torch.stack([r["post_attention_layernorm"], r["pre_feedforward_layernorm"], r["pre_feedforward_layernorm_2"]])
        ff = torch.stack([r["post_feedforward_layernorm_1"], r["post_feedforward_layernorm_2"], r["post_feedforward_layernorm"], next_norm])
        self.w_attn_c = compact_tensor(mesh, attn)
        self.w_ff_c = compact_tensor(mesh, ff)

    def _rms(self, x, w):
        return rms_norm(x, w, self.eps)

    def _finish(self, x, attn_out):
        a = self._rms(attn_out, self.post_attn_ln)
        attn_out.deallocate(True)
        x2 = ttnn.add(x, a)
        x.deallocate(True)
        a.deallocate(True)
        f = self.ff(x2)
        x3 = ttnn.add(x2, f)
        x2.deallocate(True)
        f.deallocate(True)
        if self.scalar != 1.0:
            x4 = ttnn.mul(x3, self.scalar)
            x3.deallocate(True)
            x3 = x4
        return x3

    def prefill(self, x, rope, kv, chunk):
        h = self._rms(x, self.input_ln)
        a = self.attn.prefill(h, rope, kv, chunk)
        h.deallocate(True)
        return self._finish(x, a)

    def decode_b1(self, xc, h, rope, kv, page_table, cur_pos, b: Boundary, x_page0: int = 0):
        """One decode token on compact boundaries: xc compact FP32 residual, h its normed row form.
        Returns the next (xc, h); h of the last layer is final-normed."""
        a = self.attn.decode_b1(h, rope, kv, page_table, cur_pos, b)
        g = all_gather_compact(a)
        a.deallocate(True)
        x2, n, m_in, e_in = b.compact(), b.row(), b.row(), b.row()
        b.attn_tail(g, xc, self.w_attn_c, x2, n, m_in, e_in, x_page0=x_page0)
        g.deallocate(True)
        f = self.ff.decode_b1(n, m_in, e_in, b)
        for t in (n, m_in, e_in):
            t.deallocate(True)
        g = all_gather_compact(f)
        f.deallocate(True)
        x3, h2 = b.compact(), b.row()
        b.ff_tail(g, x2, self.w_ff_c, x3, h2, self.scalar)
        g.deallocate(True)
        x2.deallocate(True)
        return x3, h2

    def decode(self, x, rope, kv, page_table, cur_pos):
        h = self._rms(x, self.input_ln)
        a = self.attn.decode(h, rope, kv, page_table, cur_pos)
        h.deallocate(True)
        return self._finish(x, a)


class Gemma4TT:
    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        ckpt: Checkpoint | None = None,
        cache_root: Path = Path("cache"),
        precision: Precision | None = None,
        max_batch: int = 4,
        max_seq_len: int = 32768,
        block_size: int = 64,
        num_layers: int | None = None,
    ):
        if max_batch > STREAM_ROWS:
            raise ValueError(f"decode streams the experts of at most {STREAM_ROWS} rows; got max_batch={max_batch}")
        self.mesh = mesh
        self.ckpt = ckpt = ckpt or Checkpoint()
        self.cfg = cfg = ckpt.config
        self.precision = precision = precision or Precision()
        self.tp = mesh.get_num_devices()
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.window = cfg.sliding_window
        self.num_layers = num_layers or cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size

        cache_dir = Path(cache_root) / f"gemma4-26b-qat-tp{self.tp}-{precision.tag()}"
        loader = WeightLoader(mesh, cache_dir)
        self.layers = [DecoderLayer(loader, ckpt, cfg, i, precision) for i in range(self.num_layers)]
        self.final_norm = loader.put("final_norm", lambda: norm_weight(ckpt.get("norm.weight")), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.lm_head = self._lm_head(loader, ckpt, precision.lm_head)
        self.lm_in = width_sharded_l1(self.hidden, LM_CORES)
        self.boundary = Boundary(mesh, self.hidden, cfg.rms_norm_eps)
        # compact tiles of each layer type's [cos | sin] table in the batch-1 FP32 input
        self._cs_tiles = {
            kind: 2 * next(layer.attn.qk_rope.V for layer in self.layers if layer.sliding == (kind == "sliding_attention"))
            for kind in ("sliding_attention", "full_attention")
        }
        final_raw = ckpt.get("norm.weight").float()
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1]._raw["input_layernorm"] if i + 1 < len(self.layers) else final_raw
            layer.boundary_weights(mesh, nxt)
        self.input_ln0 = self.layers[0]._raw["input_layernorm"]
        self.embed_table = ckpt.get("embed_tokens.weight")  # BF16 [V, H] on host
        self.embed_scale = torch.tensor(embed_scale(self.hidden), dtype=torch.bfloat16)
        # A chunk's padding can run CHUNK_ALIGN rows past the last position.
        self.rope = self._rope_tables(max_seq_len + CHUNK_ALIGN)
        self.kv: KvCache | None = None
        self.traces = {}
        self._trace_inputs: dict[int, dict[str, ttnn.Tensor]] = {}
        self._pt_last, self._pt_src = {}, {}
        init_ccl(mesh)

    # -- host-side tables ----------------------------------------------------

    def _rope_tables(self, n: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """float32 [n, hd] cos/sin per layer type; prefill uploads them as BF16, decode as fp32."""
        rotary = hf.Gemma4TextRotaryEmbedding(self.cfg)
        pos = torch.arange(n)[None]
        out = {}
        for kind in ("sliding_attention", "full_attention"):
            cos, sin = rotary(torch.zeros(1, dtype=torch.float32), pos, kind)
            out[kind] = (cos[0], sin[0])
        return out

    def _lm_head(self, loader, ckpt, dtype) -> list[ttnn.Tensor]:
        """LM_SPLITS DRAM-sharded column blocks per chip; chip c split i holds vocab
        columns c * V/tp + i * width onwards, so concatenating the splits on each chip
        and then the chips gives the vocabulary order."""
        per_chip = self.vocab // self.tp
        width = per_chip // LM_SPLITS
        mem = dram_sharded(self.mesh, self.hidden, width)
        self.lm_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.hidden // TILE // LM_CORES, per_core_M=1, per_core_N=width // TILE // LM_CORES, fused_activation=None
        )
        self.lm_out = width_sharded_l1(width, LM_CORES)
        table = {}

        def split(i):
            if "t" not in table:
                table["t"] = ckpt.get("embed_tokens.weight").T  # [H, V]
            t = table["t"]
            cols = [t[:, c * per_chip + i * width : c * per_chip + (i + 1) * width] for c in range(self.tp)]
            return torch.cat(cols, -1).contiguous()[None, None]

        heads = [loader.put(f"lm_head_ds{LM_SPLITS}_{i}", lambda i=i: split(i), dtype, shard_dim=-1, memory_config=mem) for i in range(LM_SPLITS)]
        table.clear()
        return heads

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_table[ids] * self.embed_scale  # BF16 math, as HF

    def _to_device(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh), memory_config=DRAM
        )

    # -- KV cache ------------------------------------------------------------

    def allocate_kv_cache(self, rows: int, tokens: int) -> torch.Tensor:
        """For scripts and tests without vLLM: allocate the layout vLLM would, sized
        for `rows` sequences of up to `tokens` positions, and return their page tables
        [groups, rows, width]. The adapter builds `kv` from vLLM's plan instead."""
        shapes = [layer.attn.shape for layer in self.layers]
        types = self.cfg.layer_types[: self.num_layers]
        self.use_kv(KvCache.standalone(self.mesh, types, shapes, self.block_size, self.precision.kv_cache, rows, tokens, self._ring_tokens))
        return self.page_tables(rows, tokens)

    def use_kv(self, kv: KvCache):
        """Take the KV cache, once: decode traces bind its buffers and table width."""
        if self.kv is not None:
            raise RuntimeError("the KV cache is set once per model")
        if kv.grid % CHUNK_ALIGN or PREFILL_CHUNK % kv.grid:
            raise ValueError(f"a {kv.grid}-token block grid does not fit chunked prefill's {CHUNK_ALIGN}-token SDPA chunks")
        self.kv = kv

    @property
    def table_width(self) -> int:
        """Blocks per sequence in the decode page tables (fixed, so traces can replay)."""
        # paged ops need fewer blocks per sequence than blocks in the pool
        return min(-(-self.max_seq_len // min(self.kv.block_sizes)), self.kv.num_blocks - 1)

    def page_tables(self, rows: int, tokens: int) -> torch.Tensor:
        """Disjoint page tables [groups, rows, width] for `rows` sequences of up to
        `tokens` positions, for scripts and warmup."""
        return self.kv.tables(rows, tokens, self.table_width, self._ring_tokens)

    @property
    def _ring_tokens(self) -> int:
        """Sliding positions a script's page table keeps apart: the window behind a
        prefill chunk, plus the chunk."""
        return self.window + PREFILL_CHUNK + CHUNK_ALIGN

    # -- prefill -------------------------------------------------------------

    def prefill(self, tokens: torch.Tensor, start: int, tables: torch.Tensor, all_logits: bool = False) -> torch.Tensor:
        """Compute positions [start, len(tokens)) of one sequence whose cache holds [0, start).

        tokens: the sequence from position 0. tables: int32 [groups, width], the
        sequence's page table in every KV-cache group. Returns float logits [V] of
        the last position, or [S, V] of every position from `start` with `all_logits`
        (accuracy studies).

        `start` must lie on the block grid (kv.grid): every group then writes from
        the start of a block of its own and never into a block before `start`,
        which can be a prefix-cache block that other requests read. vLLM's prefix
        hits and chunk ends are multiples of that grid.
        """
        end = tokens.shape[0]
        if not 0 <= start < end or start % self.kv.grid:
            raise ValueError(f"prefill of positions [{start}, {end}) must start on the {self.kv.grid}-token block grid")
        out = []
        for c0 in range(start, end, PREFILL_CHUNK):
            c1 = min(c0 + PREFILL_CHUNK, end)
            x = self._prefill_chunk(tokens, c0, c1, tables)
            if all_logits or c1 == end:
                lo = 0 if all_logits else c1 - 1 - c0
                last = ttnn.slice(x, [0, 0, lo, 0], [1, 1, c1 - c0, self.hidden])
                out.append(self._logits(last))
            x.deallocate(True)  # `last` can be x itself or a view of it
        return torch.cat(out) if all_logits else out[-1][-1]

    def _prefill_chunk(self, tokens, c0: int, c1: int, tables: torch.Tensor) -> ttnn.Tensor:
        """Run positions [c0, c1) through every layer, writing their K/V. Returns the
        final hidden rows, padded."""
        n = c1 - c0
        # Chunked SDPA works on whole q chunks; the first chunk only needs whole tiles.
        pad = TILE if c0 == 0 else CHUNK_ALIGN
        padded = -(-n // pad) * pad
        ids = torch.zeros(padded, dtype=torch.long)
        ids[:n] = tokens[c0:c1]
        x = self._to_device(self.embed(ids).reshape(1, 1, padded, self.hidden))
        ropes = {
            kind: (self._to_device(cos[c0 : c0 + padded][None, None]), self._to_device(sin[c0 : c0 + padded][None, None]))
            for kind, (cos, sin) in self.rope.items()
        }

        def table(t: torch.Tensor) -> ttnn.Tensor:
            # Chunked SDPA with a device chunk start reads rows of a multiple of 32 bytes.
            row = torch.zeros(1, -(-t.shape[0] // 8) * 8, dtype=torch.int32)
            row[0, : t.shape[0]] = t
            return self._to_device(row, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        fill_tables = [table(tables[g, c0 // block :]) for g, block in enumerate(self.kv.block_sizes)]
        whole = [table(tables[g]) if c0 else None for g in range(self.kv.groups)]
        start = self._to_device(torch.tensor([c0], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) if c0 else None
        chunk = PrefillChunk(c0, n, start, whole, fill_tables)
        for layer, kv in zip(self.layers, self.kv.layers):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            x = layer.prefill(x, ropes[kind], kv, chunk)
        for t in (*fill_tables, *filter(None, [*whole, start]), *[r for pair in ropes.values() for r in pair]):
            t.deallocate(True)
        return x

    # -- decode --------------------------------------------------------------

    def decode(self, tokens: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor, trace: bool = False) -> torch.Tensor:
        """One decode step for B rows. positions -1 marks inactive rows; tables is
        int32 [groups, B, width], every row's page table in each KV-cache group.

        Inactive rows are dropped before the device step: vLLM pads decode to
        max_num_seqs rows, and every row costs 8 expert slots, so one user in a
        4-row batch would otherwise pay for four. Their logits come back as zeros.
        With `trace`, the first call per active row count captures a metal trace and
        later calls only copy the inputs and replay it. Returns float logits [B, V].
        """
        rows = tokens.shape[0]
        active, logits = self.decode_active(tokens, positions, tables, trace)
        if len(active) == rows:
            return logits
        out = torch.zeros(rows, self.vocab)
        out[active] = logits
        return out

    def decode_active(self, tokens, positions, tables, trace: bool = False):
        """decode() without the inactive rows: (active row indices, float logits [A, V]).

        Building the zero-filled [B, V] result of decode() costs ~1 ms per step; the vLLM
        adapter samples only the active rows."""
        active = torch.nonzero(positions >= 0).flatten()
        if len(active) == tokens.shape[0]:
            return active, self._decode_rows(tokens, positions, tables, trace)
        if not len(active):
            return active, torch.zeros(0, self.vocab)
        return active, self._decode_rows(tokens[active], positions[active], tables[:, active], trace)

    def _decode_rows(self, tokens, positions, tables, trace) -> torch.Tensor:
        """Decode rows that are all active."""
        host = self._decode_inputs(tokens, positions, tables)
        if not trace:
            dev = {k: ttnn.to_device(v, self.mesh, memory_config=DRAM) for k, v in host.items()}
            out = self._decode_device(dev)
            logits = self._read_logits(out)
            out.deallocate(True)
            for t in dev.values():
                t.deallocate(True)
            return logits
        key = tokens.shape[0]
        if not self._trace_inputs:
            self._prepare_traces()
        if key not in self.traces:
            dev = self._trace_inputs[key]
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out = self._decode_device(dev)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[key] = (tid, dev, out)
        tid, dev, out = self.traces[key]
        # Page tables change only when a sequence gets a new block or rows move;
        # every copy costs 0.02-0.2 ms of host time, so unchanged ones are skipped.
        last = self._pt_last.setdefault(key, {})
        for k, v in host.items():
            src = self._pt_src.get(k)
            if src is not None:
                if k in last and torch.equal(last[k], src):
                    continue
                last[k] = src.clone()
            ttnn.copy_host_to_device_tensor(v, dev[k])
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        return self._read_logits(out)

    def _prepare_traces(self):
        """Allocate every batch size's trace inputs and run its decode once, eagerly,
        before the first capture.

        A buffer allocated after a capture can overlap that trace's intermediates,
        and replaying the trace then overwrites it. Constants are never rewritten,
        and unchanged page tables are skipped before a replay. A constant that an op
        creates on first use (rotate_half's matrix, a batch size's MoE masks) and a
        batch size's inputs allocated after another batch size's capture both read
        corrupted values: wrong tokens, or expert ids that send a read out of
        bounds and hang the device. The eager run also compiles the programs. Its
        zero page tables send every row's K/V write to the null block 0, which
        holds no request's data.
        """
        width = self.table_width
        for batch in range(1, self.max_batch + 1):
            for layer in self.layers:
                layer.attn.head_weights(batch)
                layer.ff.decode_constants(batch)
            zeros = torch.zeros(batch, dtype=torch.long)
            host = self._decode_inputs(zeros, zeros, torch.zeros(self.kv.groups, batch, width, dtype=torch.int32))
            self._trace_inputs[batch] = {k: ttnn.to_device(v, self.mesh, memory_config=DRAM) for k, v in host.items()}
        for dev in self._trace_inputs.values():
            self._decode_device(dev).deallocate(True)

    def _host(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _decode_inputs(self, tokens, positions, tables) -> dict[str, ttnn.Tensor]:
        """Host tensors for one decode step; shapes depend only on the batch size."""
        batch = tokens.shape[0]
        width = self.table_width
        pts = torch.zeros(self.kv.groups, batch, width, dtype=torch.int32)
        n = min(width, tables.shape[2])
        pts[:, :, :n] = tables[:, :, :n].to(torch.int32)
        rm = ttnn.ROW_MAJOR_LAYOUT
        emb = self.embed(tokens.long())
        if batch == 1:
            # compact FP32 residual and layer 0's normed input (host RMSNorm in FP32)
            xf = emb.float()
            h0 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * self.input_ln0
            first = {"h0": self._host(h0.reshape(1, 1, 1, self.hidden))}
            f32 = [to_compact(xf)[None, None]]  # one FP32 input [xc | sliding cs | full cs]: fewer host copies
        else:
            first = {"x": self._host(emb.reshape(1, 1, batch, self.hidden))}
        inputs = {**first, "cur_pos": self._host(positions.to(torch.int32), ttnn.int32, rm)}
        self._pt_src = {}
        for g in range(self.kv.groups):
            inputs[f"pt{g}"] = self._host(pts[g], ttnn.int32, rm)
            self._pt_src[f"pt{g}"] = pts[g]
        pos = positions.clamp(min=0).long()
        for kind, (cos, sin) in self.rope.items():
            attn = next(layer.attn for layer in self.layers if layer.sliding == (kind == "sliding_attention"))
            roped = attn.shape.n_q + attn.shape.n_kv  # Q and K rows; V rows keep cos 1, sin 0
            tables = {}
            for name, table, fill in (("cos", cos, 1.0), ("sin", sin, 0.0)):
                rows = torch.full((batch, attn.heads, attn.shape.head_dim), fill, dtype=table.dtype)
                rows[:, :roped] = table[pos][:, None, :]
                tables[name] = rows
            if batch == 1:  # compact [cos | signed sin] tables for the fused q/k norm + RoPE
                f32.append(attn.qk_rope.host_tables(tables["cos"].flatten(), tables["sin"].flatten()))
            else:
                for name, rows in tables.items():
                    inputs[f"{kind}.{name}"] = self._host(rows.reshape(1, 1, batch * attn.heads, -1), ttnn.float32)
        if batch == 1:
            inputs["in32"] = self._host(torch.cat(f32, dim=-1), ttnn.float32)
        return inputs

    def _decode_device(self, dev: dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """Device decode graph. Never frees its inputs, so it can be traced."""
        if "in32" in dev:  # batch 1: compact boundaries
            V = self.boundary.V
            pages = {"sliding_attention": V, "full_attention": V + self._cs_tiles["sliding_attention"]}
            xc, h = dev["in32"], dev["h0"]
            for i, (layer, kv) in enumerate(zip(self.layers, self.kv.layers)):
                kind = "sliding_attention" if layer.sliding else "full_attention"
                rope = ("compact", dev["in32"], pages[kind])
                xn, hn = layer.decode_b1(xc, h, rope, kv, dev[f"pt{kv.group}"], dev["cur_pos"], self.boundary, x_page0=0)
                if i:  # never free the step inputs
                    xc.deallocate(True)
                    h.deallocate(True)
                xc, h = xn, hn
            out = self._logits_device(h, normed=True)
            xc.deallocate(True)
            h.deallocate(True)
            return out
        x = ttnn.clone(dev["x"])
        for layer, kv in zip(self.layers, self.kv.layers):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            rope = (dev[f"{kind}.cos"], dev[f"{kind}.sin"])
            x = layer.decode(x, rope, kv, dev[f"pt{kv.group}"], dev["cur_pos"])
        out = self._logits_device(x)
        x.deallocate(True)
        return out

    # -- head ----------------------------------------------------------------

    def _logits(self, x: ttnn.Tensor) -> torch.Tensor:
        rows = x.shape[2]
        if rows > TILE:
            # ttnn.linear segfaults on the host for M > 32 against the 131072-wide
            # LM head shard (ttnn 0.81), so multi-row logits go one tile at a time.
            parts = []
            for start in range(0, rows, TILE):
                part = ttnn.slice(x, [0, 0, start, 0], [1, 1, min(start + TILE, rows), self.hidden])
                parts.append(self._logits(part))  # row slices are views of x; do not free them
            return torch.cat(parts)
        out = self._logits_device(x)
        logits = self._read_logits(out)
        out.deallocate(True)
        return logits

    def _logits_device(self, x: ttnn.Tensor, normed: bool = False) -> ttnn.Tensor:
        """Raw fp32 logits, row-major; the host applies the final softcap
        (three fp32 ops over the tile-padded [32, 65536] took 0.63 ms here)."""
        h = x if normed else rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        hs = ttnn.to_memory_config(h, self.lm_in)
        if not normed:
            h.deallocate(True)
        rows = x.shape[2]
        parts = []
        for w in self.lm_head:
            o = ttnn.linear(hs, w, program_config=self.lm_cfg, compute_kernel_config=HIFI2, memory_config=self.lm_out, dtype=ttnn.float32)
            # Untilizing straight from L1 drops the tile padding, so the host reads
            # B rows instead of 32 (9 ms -> under 1 ms for 131072 fp32 logits), and
            # the row-major concat below only moves those rows.
            parts.append(ttnn.untilize_with_unpadding(o, output_tensor_end=[0, 0, rows - 1, o.shape[3] - 1], memory_config=DRAM))
            o.deallocate(True)
        hs.deallocate(True)
        out = ttnn.concat(parts, dim=-1)
        for o in parts:
            o.deallocate(True)
        return out

    def _read_logits(self, out: ttnn.Tensor) -> torch.Tensor:
        logits = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
        cap = self.cfg.final_logit_softcapping
        logits = logits.reshape(-1, self.vocab)
        if cap:
            # numpy ufuncs run on this thread. torch's pool sleeps during the ~30 ms
            # device wait, and waking it made the same in-place softcap take 1 ms
            # instead of 0.2 ms.
            arr = logits.numpy()
            np.divide(arr, cap, out=arr)
            np.tanh(arr, out=arr)
            np.multiply(arr, cap, out=arr)
        return logits

    def release_traces(self):
        """Release the captures; their inputs stay allocated for the next capture."""
        for tid, _, _ in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}
        self._pt_last = {}
