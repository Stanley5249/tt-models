"""Gemma 4 26B-A4B text model on a 1x2 Wormhole mesh.

The token embedding stays on the host (1.5 GB BF16 that would otherwise take
device memory); the tied LM head is uploaded once, split over the vocabulary.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn
from transformers.models.gemma4 import modeling_gemma4 as hf

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.reference import embed_scale
from gemma4_tt.tt.attention import Attention, ring_selection
from gemma4_tt.tt.common import DRAM, HIFI2, HIFI4, TILE, WeightLoader, norm_weight, rms_norm
from gemma4_tt.tt.moe import FeedForward


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

    def prefill(self, x, rope, kv_cache, fill):
        h = self._rms(x, self.input_ln)
        a = self.attn.prefill(h, rope, kv_cache, fill)
        h.deallocate(True)
        return self._finish(x, a)

    def decode(self, x, rope, kv_cache, page_table, cur_pos):
        h = self._rms(x, self.input_ln)
        a = self.attn.decode(h, rope, kv_cache, page_table, cur_pos)
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
        self.lm_head = loader.put(
            "lm_head", lambda: ckpt.get("embed_tokens.weight").T.contiguous()[None, None], precision.lm_head, shard_dim=-1
        )
        self.embed_table = ckpt.get("embed_tokens.weight")  # BF16 [V, H] on host
        self.embed_scale = torch.tensor(embed_scale(self.hidden), dtype=torch.bfloat16)
        self.rope = self._rope_tables(max_seq_len)
        self.kv_caches = None
        self.traces = {}

    # -- host-side tables ----------------------------------------------------

    def _rope_tables(self, n: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        rotary = hf.Gemma4TextRotaryEmbedding(self.cfg)
        pos = torch.arange(n)[None]
        out = {}
        for kind in ("sliding_attention", "full_attention"):
            cos, sin = rotary(torch.zeros(1, dtype=torch.float32), pos, kind)
            out[kind] = (cos[0].to(torch.bfloat16), sin[0].to(torch.bfloat16))  # [n, hd]
        return out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_table[ids] * self.embed_scale  # BF16 math, as HF

    def _to_device(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh), memory_config=DRAM
        )

    # -- KV cache ------------------------------------------------------------

    def allocate_kv_cache(self, num_blocks: int):
        """Global layers: [num_blocks, kv/chip, block, 512]. Sliding layers: one ring of `window` per slot."""
        bs = self.block_size
        self.num_blocks = num_blocks
        ring_blocks = self.window // bs
        caches = []
        for layer in self.layers:
            s = layer.attn.shape
            blocks = self.max_batch * ring_blocks if s.sliding else num_blocks
            shape = (blocks, s.n_kv, bs, s.head_dim)
            pair = [
                ttnn.zeros(shape, dtype=self.precision.kv_cache, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM)
                for _ in range(2)
            ]
            caches.append(pair)
        self.kv_caches = caches
        self.slot_blocks = torch.arange(self.max_batch * ring_blocks, dtype=torch.int32).reshape(self.max_batch, ring_blocks)
        return caches

    def remap_slots(self, slot_remap):
        """Destination slot i takes the sliding ring of old slot slot_remap[i] (host-side only)."""
        self.slot_blocks = self.slot_blocks[torch.as_tensor(slot_remap, dtype=torch.long)]

    # -- prefill -------------------------------------------------------------

    def prefill(self, tokens: torch.Tensor, page_table: torch.Tensor, slot: int, all_logits: bool = False) -> torch.Tensor:
        """Prefill one request from position 0. Returns float logits [V] of the last token
        (or [S, V] for every position with `all_logits`, used by accuracy studies).

        page_table: int32 [num_blocks_for_request] vLLM block ids for the global layers.
        slot: the batch slot whose sliding ring receives this request.
        """
        seq_len = tokens.shape[0]
        padded = -(-seq_len // TILE) * TILE
        if padded > self.window and padded < self.window + TILE:
            padded = self.window + TILE  # the ring tail needs window + 32 rows
        ids = torch.zeros(padded, dtype=torch.long)
        ids[:seq_len] = tokens
        x = self._to_device(self.embed(ids).reshape(1, 1, padded, self.hidden))

        ropes = {}
        for kind, (cos, sin) in self.rope.items():
            ropes[kind] = (self._to_device(cos[:padded][None, None]), self._to_device(sin[:padded][None, None]))

        need_blocks = -(-padded // self.block_size)
        pt = page_table[:need_blocks].to(torch.int32)
        if pt.shape[0] < need_blocks:
            raise ValueError(f"page table has {pt.shape[0]} blocks, prefill of {padded} tokens needs {need_blocks}")
        global_pt = self._to_device(pt[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ring_pt = self._to_device(self.slot_blocks[slot : slot + 1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        sel = ring_selection(seq_len, padded, self.window)
        sel_tt = None if sel is None else (sel[0], self._to_device(sel[1]))
        kv_dtype = self.precision.kv_cache

        def fill_global(k, v, cache):
            for t, c in zip((k, v), cache):
                t8 = ttnn.typecast(t, kv_dtype)
                ttnn.experimental.paged_fill_cache(c, t8, global_pt, batch_idx=0)
                t8.deallocate(True)

        def fill_sliding(k, v, cache):
            for t, c in zip((k, v), cache):
                if sel_tt is None:
                    rows = t
                else:
                    start, s_tt = sel_tt
                    # A slice covering the whole tensor returns the input itself; never free k/v.
                    tail = t if start == 0 else ttnn.slice(t, [0, 0, start, 0], [1, t.shape[1], padded, t.shape[3]])
                    rows = ttnn.matmul(s_tt, tail, compute_kernel_config=HIFI4, memory_config=DRAM)
                    if tail is not t:
                        tail.deallocate(True)
                t8 = ttnn.typecast(rows, kv_dtype)
                if rows is not t:
                    rows.deallocate(True)
                ttnn.experimental.paged_fill_cache(c, t8, ring_pt, batch_idx=0)
                t8.deallocate(True)

        for layer, cache in zip(self.layers, self.kv_caches):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            fill = fill_sliding if layer.sliding else fill_global
            x = layer.prefill(x, ropes[kind], cache, fill)

        first = 0 if all_logits else seq_len - 1
        last = ttnn.slice(x, [0, 0, first, 0], [1, 1, seq_len, self.hidden])
        logits = self._logits(last)
        x.deallocate(True)  # `last` can be x itself or a view of it
        for t in (global_pt, ring_pt, *[r for pair in ropes.values() for r in pair]):
            t.deallocate(True)
        if sel_tt is not None:
            sel_tt[1].deallocate(True)
        return logits if all_logits else logits[0]

    # -- decode --------------------------------------------------------------

    def decode(self, tokens: torch.Tensor, positions: torch.Tensor, page_table: torch.Tensor, trace: bool = False) -> torch.Tensor:
        """One decode step for B rows (row i is slot i). positions -1 marks inactive rows.

        With `trace`, the first call per batch size captures a metal trace and
        later calls only copy the inputs and replay it. Returns float logits [B, V].
        """
        host = self._decode_inputs(tokens, positions, page_table)
        if not trace:
            dev = {k: ttnn.to_device(v, self.mesh, memory_config=DRAM) for k, v in host.items()}
            out = self._decode_device(dev)
            logits = self._read_logits(out)
            out.deallocate(True)
            for t in dev.values():
                t.deallocate(True)
            return logits
        key = tokens.shape[0]
        if key not in self.traces:
            dev = {k: ttnn.to_device(v, self.mesh, memory_config=DRAM) for k, v in host.items()}
            self._decode_device(dev).deallocate(True)  # compile outside the capture
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out = self._decode_device(dev)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[key] = (tid, dev, out)
        tid, dev, out = self.traces[key]
        for k, v in host.items():
            ttnn.copy_host_to_device_tensor(v, dev[k])
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        return self._read_logits(out)

    def _host(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _decode_inputs(self, tokens, positions, page_table) -> dict[str, ttnn.Tensor]:
        """Host tensors for one decode step; shapes depend only on the batch size."""
        batch = tokens.shape[0]
        # paged ops need fewer blocks per sequence than blocks in the pool
        width = min(self.max_seq_len // self.block_size, self.num_blocks - 1)
        pt = torch.zeros(batch, width, dtype=torch.int32)
        n = min(width, page_table.shape[1])
        pt[:, :n] = page_table[:, :n].to(torch.int32)
        rm = ttnn.ROW_MAJOR_LAYOUT
        inputs = {
            "x": self._host(self.embed(tokens.long()).reshape(1, 1, batch, self.hidden)),
            "cur_pos": self._host(positions.to(torch.int32), ttnn.int32, rm),
            "global_pt": self._host(pt, ttnn.int32, rm),
            "ring_pt": self._host(self.slot_blocks[:batch].contiguous(), ttnn.int32, rm),
        }
        pos = positions.clamp(min=0).long()
        for kind, (cos, sin) in self.rope.items():
            attn = next(layer.attn for layer in self.layers if layer.sliding == (kind == "sliding_attention"))
            roped = attn.shape.n_q + attn.shape.n_kv  # Q and K rows; V rows keep cos 1, sin 0
            for name, table, fill in (("cos", cos, 1.0), ("sin", sin, 0.0)):
                rows = torch.full((batch, attn.heads, attn.shape.head_dim), fill, dtype=table.dtype)
                rows[:, :roped] = table[pos][:, None, :]
                inputs[f"{kind}.{name}"] = self._host(rows.reshape(1, 1, batch * attn.heads, -1))
        return inputs

    def _decode_device(self, dev: dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """Device decode graph. Never frees its inputs, so it can be traced."""
        x = ttnn.clone(dev["x"])
        for layer, cache in zip(self.layers, self.kv_caches):
            kind = "sliding_attention" if layer.sliding else "full_attention"
            rope = (dev[f"{kind}.cos"], dev[f"{kind}.sin"])
            pt = dev["ring_pt"] if layer.sliding else dev["global_pt"]
            x = layer.decode(x, rope, cache, pt, dev["cur_pos"])
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

    def _logits_device(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        logits = ttnn.linear(h, self.lm_head, compute_kernel_config=HIFI2, memory_config=DRAM, dtype=ttnn.float32)
        h.deallocate(True)
        cap = self.cfg.final_logit_softcapping
        scaled = ttnn.mul(logits, 1.0 / cap)
        logits.deallocate(True)
        t = ttnn.tanh(scaled)
        scaled.deallocate(True)
        capped = ttnn.mul(t, cap)
        t.deallocate(True)
        # Row-major drops the tile padding, so the host reads B rows instead of
        # 32 (9 ms -> under 1 ms for one user's 131072 fp32 logits).
        out = ttnn.to_layout(capped, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        capped.deallocate(True)
        return out

    def _read_logits(self, out: ttnn.Tensor) -> torch.Tensor:
        logits = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
        return logits.reshape(-1, self.vocab)

    def release_traces(self):
        for tid, dev, out in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}
