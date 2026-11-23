"""Gemma 4 26B-A4B text model on a 1x2 Wormhole mesh.

The token embedding stays on the host (1.5 GB BF16 that would otherwise take
device memory); the tied LM head is uploaded once, split over the vocabulary.

The model holds no per-request state: every K/V row lives in the paged cache
(kv_cache.py) at the blocks the caller's page tables name, so requests can be
moved, preempted, resumed and share cached prefixes without telling the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import ttnn
from transformers.models.gemma4 import modeling_gemma4 as hf

from tt_models.common.ccl import init_ccl
from tt_models.common.config import DRAM
from tt_models.common.weights import WeightLoader
from tt_models.models.gemma4.attention import CHUNK_ALIGN, PrefillChunk
from tt_models.models.gemma4.boundary import Boundary
from tt_models.models.gemma4.checkpoint import Checkpoint, WeightSource
from tt_models.models.gemma4.compact import to_compact
from tt_models.models.gemma4.config import (
    PREFILL_BUCKET,
    PREFILL_CHUNK,
    STREAM_ROWS,
    Precision,
    prefill_rows,
)
from tt_models.models.gemma4.kv_cache import KvCache
from tt_models.models.gemma4.layer import DecoderLayer
from tt_models.models.gemma4.lm_head import LmHead
from tt_models.models.gemma4.reference import embed_scale

if TYPE_CHECKING:
    from collections.abc import Sequence


class Gemma4TT:
    """Run Gemma 4 prefill and decode on a Tenstorrent mesh."""

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        ckpt: WeightSource | None = None,
        cache_root: Path = Path("cache"),
        precision: Precision | None = None,
        max_batch: int = 4,
        max_seq_len: int = 32768,
        block_size: int = 64,
        num_layers: int | None = None,
    ) -> None:
        """Load the model weights and initialize the device operators."""
        if max_batch > STREAM_ROWS:
            raise ValueError(
                f"decode streams the experts of at most {STREAM_ROWS} rows; got "
                f"max_batch={max_batch}"
            )
        if precision is not None and precision.expert_q4 and max_batch > 1:
            raise ValueError(
                "the experimental Q4_0 experts decode one row; build the model with "
                "max_batch=1"
            )
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
        self.layers = [
            DecoderLayer(loader, ckpt, cfg, i, precision)
            for i in range(self.num_layers)
        ]
        self.head = LmHead(mesh, loader, ckpt, cfg, precision.lm_head)
        self.boundary = Boundary(mesh, self.hidden, cfg.rms_norm_eps)
        # compact tiles per row of each layer type's [cos | sin] table in the FP32 input
        self._cs_tiles = {
            kind: 2
            * next(
                layer.attn.qk_rope.V
                for layer in self.layers
                if layer.sliding == (kind == "sliding_attention")
            )
            for kind in ("sliding_attention", "full_attention")
        }
        final_raw = ckpt.get("norm.weight").float()
        for i, layer in enumerate(self.layers):
            nxt = (
                self.layers[i + 1].input_norm if i + 1 < len(self.layers) else final_raw
            )
            layer.boundary_weights(mesh, nxt)
        self.input_ln0 = self.layers[0].input_norm
        self.embed_table = ckpt.get("embed_tokens.weight")  # BF16 [V, H] on host
        self.embed_scale = torch.tensor(embed_scale(self.hidden), dtype=torch.bfloat16)
        # A piece's padding can run PREFILL_BUCKET rows past the last position.
        self.rope = self._rope_tables(max_seq_len + PREFILL_BUCKET)
        self.kv: KvCache | None = None
        self.traces: dict[
            int, tuple[ttnn.MeshTraceId, dict[str, ttnn.Tensor], ttnn.Tensor]
        ] = {}
        self._trace_inputs: dict[int, dict[str, ttnn.Tensor]] = {}
        self._pt_last: dict[int, dict[str, torch.Tensor]] = {}
        self._pt_src: dict[str, torch.Tensor] = {}
        init_ccl(mesh)

    # -- host-side tables ----------------------------------------------------

    def _rope_tables(self, n: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Build cosine and sine tables per layer type.

        float32 [n, hd] cos/sin per layer type; prefill uploads them as BF16, decode
        as fp32.
        """
        rotary = hf.Gemma4TextRotaryEmbedding(self.cfg)
        pos = torch.arange(n)[None]
        out = {}
        for kind in ("sliding_attention", "full_attention"):
            cos, sin = rotary(torch.zeros(1, dtype=torch.float32), pos, kind)
            out[kind] = (cos[0], sin[0])
        return out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings on the host."""
        return self.embed_table[ids] * self.embed_scale  # BF16 math, as HF

    def _to_device(
        self,
        t: torch.Tensor,
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=DRAM,
        )

    # -- KV cache ------------------------------------------------------------

    def allocate_kv_cache(self, rows: int, tokens: int) -> torch.Tensor:
        """Allocate a standalone KV cache and return its page tables.

        For scripts and tests without vLLM: allocate the layout vLLM would, sized
        for `rows` sequences of up to `tokens` positions, and return their page tables
        [groups, rows, width]. The adapter builds `kv` from vLLM's plan instead.
        """
        shapes = [layer.attn.shape for layer in self.layers]
        assert self.cfg.layer_types is not None
        types = self.cfg.layer_types[: self.num_layers]
        self.use_kv(
            KvCache.standalone(
                self.mesh,
                types,
                shapes,
                self.block_size,
                self.precision.kv_cache,
                rows,
                tokens,
                self._ring_tokens,
            )
        )
        return self.page_tables(rows, tokens)

    def use_kv(self, kv: KvCache) -> None:
        """Take the KV cache, once: decode traces bind its buffers and table width."""
        if self.kv is not None:
            raise RuntimeError("the KV cache is set once per model")
        if (
            kv.grid % CHUNK_ALIGN
            or PREFILL_CHUNK % kv.grid
            or PREFILL_BUCKET % CHUNK_ALIGN
        ):
            raise ValueError(
                f"a {kv.grid}-token block grid does not fit chunked prefill's "
                f"{CHUNK_ALIGN}-token SDPA chunks"
            )
        self.kv = kv

    def require_kv(self) -> KvCache:
        """Return the installed KV cache before running a model step."""
        kv = self.kv
        if kv is None:
            raise RuntimeError("the KV cache must be installed first")
        return kv

    @property
    def table_width(self) -> int:
        """Return the fixed decode page table width so traces can replay."""
        # paged ops need fewer blocks per sequence than blocks in the pool
        kv = self.require_kv()
        return min(-(-self.max_seq_len // min(kv.block_sizes)), kv.num_blocks - 1)

    def page_tables(self, rows: int, tokens: int) -> torch.Tensor:
        """Build disjoint page tables for standalone use.

        Disjoint page tables [groups, rows, width] for `rows` sequences of up to
        `tokens` positions, for scripts and warmup.
        """
        return self.require_kv().tables(
            rows, tokens, self.table_width, self._ring_tokens
        )

    @property
    def _ring_tokens(self) -> int:
        """Return the sliding positions that must use separate cache slots.

        Sliding positions a script's page table keeps apart: the window behind a
        prefill chunk, plus the chunk.
        """
        return self.window + PREFILL_CHUNK + CHUNK_ALIGN

    # -- prefill -------------------------------------------------------------

    def prefill(
        self,
        tokens: torch.Tensor,
        start: int,
        tables: torch.Tensor,
        *,
        all_logits: bool = False,
    ) -> torch.Tensor:
        """Compute a suffix of one sequence using the cached prefix.

        Compute positions [start, len(tokens)) of one sequence whose cache holds [0,
        start).

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
        self._check_start(start, end)
        out = []
        for c0 in range(start, end, PREFILL_CHUNK):
            c1 = min(c0 + PREFILL_CHUNK, end)
            x = self._prefill_pieces([(tokens, c0, c1, tables)])
            if all_logits or c1 == end:
                lo = 0 if all_logits else c1 - 1 - c0
                last = ttnn.slice(x, [0, 0, lo, 0], [1, 1, c1 - c0, self.hidden])
                out.append(self.head.logits(last))
            x.deallocate(True)  # `last` can be x itself or a view of it
        return torch.cat(out) if all_logits else out[-1][-1]

    def prefill_batch(
        self, pieces: Sequence[tuple[torch.Tensor, int, torch.Tensor]]
    ) -> torch.Tensor:
        """Prefill several sequences in one pass; logits [n, V] of their last rows.

        pieces: (tokens, start, tables) as prefill() takes them, each computing
        positions [start, len(tokens)) in one piece, with at most PREFILL_CHUNK padded
        rows in all (batch_rows). The layers' projections, MLP and experts run on
        every piece's rows at once, so their weights are read once for all pieces:
        fifteen 144-token prompts took 3.1 s one by one.
        """
        spans: list[tuple[torch.Tensor, int, int, torch.Tensor]] = []
        for tokens, start, tables in pieces:
            self._check_start(start, tokens.shape[0])
            spans.append((tokens, start, tokens.shape[0], tables))
        if self.batch_rows([(t.shape[0], s) for t, s, _ in pieces]) > PREFILL_CHUNK:
            raise ValueError(f"a batched prefill holds at most {PREFILL_CHUNK} rows")
        x = self._prefill_pieces(spans)
        lasts: list[ttnn.Tensor] = []
        row0 = 0
        for _, c0, c1, _ in spans:
            last = row0 + c1 - c0 - 1
            lasts.append(ttnn.slice(x, [0, 0, last, 0], [1, 1, last + 1, self.hidden]))
            row0 += prefill_rows(c1 - c0, c0)
        rows = ttnn.concat(lasts, dim=2) if len(lasts) > 1 else lasts[0]
        logits = self.head.logits(rows)
        x.deallocate(True)  # the slices can be views of x
        for t in {id(t): t for t in (*lasts, rows)}.values():
            t.deallocate(True)
        return logits

    @staticmethod
    def batch_rows(pieces: Sequence[tuple[int, int]]) -> int:
        """Padded rows of a batched prefill of (end, start) pieces."""
        return sum(prefill_rows(end - start, start) for end, start in pieces)

    def _check_start(self, start: int, end: int) -> None:
        grid = self.require_kv().grid
        if not 0 <= start < end or start % grid:
            raise ValueError(
                f"prefill of positions [{start}, {end}) must start on the "
                f"{grid}-token block grid"
            )

    def _prefill_pieces(
        self, pieces: Sequence[tuple[torch.Tensor, int, int, torch.Tensor]]
    ) -> ttnn.Tensor:
        """Run prefill pieces through every layer and write their K/V.

        pieces: (tokens, c0, c1, tables), positions [c0, c1) of a sequence, each
        padded to prefill_rows and laid one after another. Returns the final hidden
        rows, padded.
        """
        cache = self.require_kv()

        def table(t: torch.Tensor, width: int) -> ttnn.Tensor:
            # Chunked SDPA with a device chunk start reads rows of a multiple of 32
            # bytes.
            # Every table keeps the full width, so a new chunk offset builds no new
            # programs.
            row = torch.zeros(1, -(-width // 8) * 8, dtype=torch.int32)
            row[0, : t.shape[0]] = t
            return self._to_device(row, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        ids: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        chunks: list[PrefillChunk] = []
        inputs: list[ttnn.Tensor] = []
        row0 = 0
        for tokens, c0, c1, tables in pieces:
            n = c1 - c0
            padded = prefill_rows(n, c0)
            piece = torch.zeros(padded, dtype=torch.long)
            piece[:n] = tokens[c0:c1]
            ids.append(piece)
            positions.append(torch.arange(c0, c0 + padded))
            width = tables.shape[1]
            fill_tables = [
                table(tables[g, c0 // block :], width)
                for g, block in enumerate(cache.block_sizes)
            ]
            whole = [
                table(tables[g], width) if c0 else None for g in range(cache.groups)
            ]
            start = (
                self._to_device(
                    torch.tensor([c0], dtype=torch.int32),
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                if c0
                else None
            )
            chunks.append(PrefillChunk(c0, n, row0, padded, start, whole, fill_tables))
            inputs += [*fill_tables, *(t for t in (*whole, start) if t is not None)]
            row0 += padded
        ids_all, pos = torch.cat(ids), torch.cat(positions)
        x = self._to_device(self.embed(ids_all).reshape(1, 1, row0, self.hidden))
        ropes = {
            kind: (
                self._to_device(cos[pos][None, None]),
                self._to_device(sin[pos][None, None]),
            )
            for kind, (cos, sin) in self.rope.items()
        }
        for layer, kv in zip(self.layers, cache.layers):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            x = layer.prefill(x, ropes[kind], kv, chunks)
        for t in (*inputs, *[r for pair in ropes.values() for r in pair]):
            t.deallocate(True)
        return x

    # -- decode --------------------------------------------------------------

    def decode(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        *,
        trace: bool = False,
    ) -> torch.Tensor:
        """Decode a batch and return logits in the original row order.

        One decode step for B rows. positions -1 marks inactive rows; tables is
        int32 [groups, B, width], every row's page table in each KV-cache group.

        Inactive rows are dropped before the device step: vLLM pads decode to
        max_num_seqs rows, and every row costs 8 expert slots, so one user in a
        4-row batch would otherwise pay for four. Their logits come back as zeros.
        With `trace`, the first call per active row count captures a metal trace and
        later calls only copy the inputs and replay it. Returns float logits [B, V].
        """
        rows = tokens.shape[0]
        active, logits = self.decode_active(tokens, positions, tables, trace=trace)
        if len(active) == rows:
            return logits
        out = torch.zeros(rows, self.vocab)
        out[active] = logits
        return out

    def decode_active(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        *,
        trace: bool = False,
        softcap: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode only the active rows and return their indices and logits.

        decode() without the inactive rows: (active row indices, float logits [A,
        V]).

        Building the zero-filled [B, V] result of decode() costs ~1 ms per step; the
        vLLM
        adapter samples only the active rows. softcap=False returns the logits before
        the final softcap (LmHead.softcap), for a caller that caps only what it needs.
        """
        active = torch.nonzero(positions >= 0).flatten()
        if len(active) == tokens.shape[0]:
            return active, self._decode_rows(
                tokens, positions, tables, trace=trace, softcap=softcap
            )
        if not len(active):
            return active, torch.zeros(0, self.vocab)
        return active, self._decode_rows(
            tokens[active],
            positions[active],
            tables[:, active],
            trace=trace,
            softcap=softcap,
        )

    def _decode_rows(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        *,
        trace: bool,
        softcap: bool = True,
    ) -> torch.Tensor:
        """Decode rows that are all active."""
        if tokens.shape[0] > self.max_batch:
            raise ValueError(
                f"{tokens.shape[0]} active decode rows; the model was built for "
                f"{self.max_batch}"
            )
        host = self._decode_inputs(tokens, positions, tables)
        if not trace:
            dev = {
                k: ttnn.to_device(v, self.mesh, memory_config=DRAM)
                for k, v in host.items()
            }
            out = self._decode_device(dev)
            logits = self.head.read(out, softcap=softcap)
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
        return self.head.read(out, softcap=softcap)

    def _prepare_traces(self) -> None:
        """Prepare one decode trace for every active row count.

        Allocate every batch size's trace inputs and run its decode once, eagerly,
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
                layer.ff.decode_constants(batch)
            zeros = torch.zeros(batch, dtype=torch.long)
            host = self._decode_inputs(
                zeros,
                zeros,
                torch.zeros(self.require_kv().groups, batch, width, dtype=torch.int32),
            )
            self._trace_inputs[batch] = {
                k: ttnn.to_device(v, self.mesh, memory_config=DRAM)
                for k, v in host.items()
            }
        for dev in self._trace_inputs.values():
            self._decode_device(dev).deallocate(True)

    def _host(
        self,
        t: torch.Tensor,
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _decode_inputs(
        self, tokens: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor
    ) -> dict[str, ttnn.Tensor]:
        """Host tensors for one decode step; shapes depend only on the batch size.

        Built on one torch thread: torch's pool sleeps while the host waits for the
        device, and waking it for these small tensors cost 1-2 ms per step (2.2 ->
        0.6 ms at 4 rows). The sampler's large argmax keeps the pool.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._build_decode_inputs(tokens, positions, tables)
        finally:
            torch.set_num_threads(threads)

    def _build_decode_inputs(
        self, tokens: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor
    ) -> dict[str, ttnn.Tensor]:
        batch = tokens.shape[0]
        width = self.table_width
        cache = self.require_kv()
        pts = torch.zeros(cache.groups, batch, width, dtype=torch.int32)
        n = min(width, tables.shape[2])
        pts[:, :, :n] = tables[:, :, :n].to(torch.int32)
        rm = ttnn.ROW_MAJOR_LAYOUT
        emb = self.embed(tokens.long())
        # compact FP32 residuals and layer 0's normed input (host RMSNorm in FP32)
        xf = emb.float()
        h0 = (
            xf
            * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
            * self.input_ln0
        )
        # one FP32 input [xc | sliding cs | full cs]: fewer host copies
        f32 = [to_compact(xf)[None, None]]
        inputs = {
            "h0": self._host(h0.reshape(1, 1, batch, self.hidden)),
            "cur_pos": self._host(positions.to(torch.int32), ttnn.int32, rm),
        }
        self._pt_src = {}
        for g in range(cache.groups):
            inputs[f"pt{g}"] = self._host(pts[g], ttnn.int32, rm)
            self._pt_src[f"pt{g}"] = pts[g]
        pos = positions.clamp(min=0).long()
        for kind, (cos, sin) in self.rope.items():
            attn = next(
                layer.attn
                for layer in self.layers
                if layer.sliding == (kind == "sliding_attention")
            )
            roped = (
                attn.shape.n_q + attn.shape.n_kv
            )  # Q and K rows; V rows keep cos 1, sin 0
            rope_tables: dict[str, torch.Tensor] = {}
            for name, table, fill in (("cos", cos, 1.0), ("sin", sin, 0.0)):
                rows = torch.full(
                    (batch, attn.heads, attn.shape.head_dim), fill, dtype=table.dtype
                )
                rows[:, :roped] = table[pos][:, None, :]
                rope_tables[name] = rows
            # compact [cos | signed sin] tables for the fused q/k norm + RoPE
            f32.append(
                attn.qk_rope.host_tables(
                    rope_tables["cos"].reshape(batch, -1),
                    rope_tables["sin"].reshape(batch, -1),
                )
            )
        inputs["in32"] = self._host(torch.cat(f32, dim=-1), ttnn.float32)
        return inputs

    def _decode_device(self, dev: dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """Device decode graph. Never frees its inputs, so it can be traced."""
        # Compact boundaries at every batch size (boundary.py).
        batch = dev["h0"].shape[2]
        V = self.boundary.V
        pages = {
            "sliding_attention": batch * V,
            "full_attention": batch * (V + self._cs_tiles["sliding_attention"]),
        }
        xc, h = dev["in32"], dev["h0"]
        for i, (layer, kv) in enumerate(zip(self.layers, self.require_kv().layers)):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            # the fused q/k norm + RoPE op reads its tables from in32
            rope = ("compact", dev["in32"], pages[kind])
            xn, hn = layer.decode_compact(
                xc,
                h,
                rope,
                kv,
                dev[f"pt{kv.group}"],
                dev["cur_pos"],
                self.boundary,
                x_page0=0,
            )
            if i:  # never free the step inputs
                xc.deallocate(True)
                h.deallocate(True)
            xc, h = xn, hn
        out = self.head.device(h, normed=True)
        xc.deallocate(True)
        h.deallocate(True)
        return out

    def release_traces(self) -> None:
        """Release the captures; their inputs stay allocated for the next capture."""
        for tid, _, _ in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces: dict[
            int, tuple[ttnn.MeshTraceId, dict[str, ttnn.Tensor], ttnn.Tensor]
        ] = {}
        self._pt_last = {}
