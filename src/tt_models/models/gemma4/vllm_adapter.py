"""vLLM TT plugin adapter for Gemma 4 26B-A4B on an N300.

Register it for a launch with

    TT_MODEL_CLASS_OVERRIDES=TTGemma4ForConditionalGeneration=tt_models.models.gemma4.vllm_adapter:Gemma4ForCausalLM

vLLM pages all 30 layers through its hybrid KV-cache manager: the sliding
layers as SlidingWindowSpec, the global ones as FullAttentionSpec (layout in
kv_cache.py). The model keeps no per-request state, so prefix
caching, chunked prefill, preemption and row moves need no adapter hooks:
release_request and slot_remap have nothing to act on.

Prefill starts on the KV block grid (128 tokens with --block-size 64), where
vLLM's prefix hits and the plugin's chunk ends fall. vLLM's Python
streaming-input sessions can resume off that grid and are not supported.
"""

import math
import os
import time

import torch
from loguru import logger

from tt_models.models.gemma4.checkpoint import Checkpoint, attention_dims
from tt_models.models.gemma4.config import PREFILL_BUCKET, PREFILL_CHUNK, prefill_rows
from tt_models.models.gemma4.kv_cache import KvCache, kv_groups
from tt_models.models.gemma4.model import Gemma4TT

# Device DRAM per chip for the shared KV pool. Weights take 8.0 of 11.87 GiB;
# the rest holds prefill activations, decode traces and their inputs.
# GEMMA4_KV_POOL_GIB overrides it, for example to force preemption in a test.
KV_POOL_BYTES = int(float(os.environ.get("GEMMA4_KV_POOL_GIB", "2.5")) * 2**30)
BFP8_BYTES = (
    1088 / 1024
)  # per value: BFP8 tiles carry one shared exponent per 16 values

# Largest top-k the model-side sampler takes; requests with no top-k (or a
# larger one) and a temperature fall back to vLLM's host sampler.
MAX_TOP_K = 1024
LOGPROBS_K = 32


def sample(
    logits: torch.Tensor, params, positions: torch.Tensor, rows_idx=None, softcap=None
):
    """Sample one token per row from float logits [B, V] with vLLM's semantics.

    rows_idx: the batch rows these logits belong to (their sampling params); default 0..B-1.
    softcap: the model's in-place final softcap when `logits` are taken before it.
    The softcap keeps the order, so greedy and top-k work on the raw logits and only
    the k candidates are capped; logprobs need the whole capped row. (Only raw
    logits above about 270, where fp32 tanh saturates, tie after capping; the raw
    order then picks the truly larger one.)

    vLLM's CPU sampler sorts the whole 262k vocabulary whenever top-p is set
    (14 ms per row on one thread, which vLLM's worker uses). Top-k first and
    top-p inside the k candidates gives the same distribution for k <= MAX_TOP_K
    in about 0.5 ms. Temperature 0 is greedy. A seeded row draws from a
    generator seeded by (seed, position), so it is reproducible.

    Returns token ids [B, 1] and, when any row asked for logprobs, the
    (top logprobs, token ids) pair of the raw logits for the plugin to pack.
    """
    rows = logits.shape[0]
    idx = list(range(rows)) if rows_idx is None else [int(r) for r in rows_idx]
    temps = [float(params.temperature[r]) for r in idx]
    want_log_probs = params.enable_log_probs is not None and any(
        params.enable_log_probs[r] for r in idx
    )
    if all(t == 0.0 for t in temps) and not want_log_probs:
        return logits.argmax(-1, keepdim=True), None  # greedy: no top-k sort
    if want_log_probs and softcap is not None:
        logits, softcap = softcap(logits), None
    top_ks = [
        int(params.top_k[r]) if int(params.top_k[r]) > 0 else MAX_TOP_K for r in idx
    ]
    k_max = min(max(top_ks + [1]), MAX_TOP_K)
    vals, ids = logits.topk(k_max, dim=-1)  # sorted, largest first
    if softcap is not None:
        vals = softcap(vals.contiguous())
    out = ids[:, :1].clone()
    for i, r in enumerate(idx):
        if temps[i] == 0.0:
            continue
        k = min(top_ks[i], k_max)
        probs = torch.softmax(vals[i, :k] / temps[i], dim=-1)
        top_p = float(params.top_p[r])
        if top_p < 1.0:
            before = torch.cumsum(probs, 0) - probs  # mass of the better candidates
            probs = torch.where(before < top_p, probs, torch.zeros_like(probs))
        seed = params.seed[r] if params.seed is not None else None
        gen = (
            None
            if seed is None
            else torch.Generator().manual_seed(
                (int(seed) * 1_000_003 + int(positions[i])) % 2**63
            )
        )
        out[i, 0] = ids[i, torch.multinomial(probs, 1, generator=gen)]
    log_probs = None
    if want_log_probs:
        lp, lp_ids = torch.log_softmax(logits, dim=-1).topk(LOGPROBS_K, dim=-1)
        log_probs = (lp, lp_ids)
    return out, log_probs


class Gemma4ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
        "supports_async_decode": False,
        # "Device" sampling here is the host-side top-k sampler above, which the
        # plugin calls through decode_forward(sampling_params=...).
        "supports_sample_on_device": True,
        "max_device_top_k": MAX_TOP_K,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
    }

    # The longest prefill piece the scheduler sends, set by get_kv_cache_spec (the
    # plugin passes no scheduler settings to the model itself).
    longest_prefill = PREFILL_CHUNK

    def __init__(self, model: Gemma4TT):
        self.model = model
        self._prefill_warmed = False
        self.model_capabilities = dict(type(self).model_capabilities)
        # GEMMA4_TIMING=1 logs the model's share of every step, to separate it from vLLM's.
        self.timing = os.environ.get("GEMMA4_TIMING") == "1"
        self._last_return = None

    # -- setup ---------------------------------------------------------------

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **_,
    ):
        if tt_data_parallel != 1:
            raise ValueError(
                "Gemma 4 26B-A4B on N300 uses one 1x2 mesh; data parallelism is not supported"
            )
        name = getattr(hf_config, "name_or_path", None) or getattr(
            hf_config, "_name_or_path", None
        )
        ckpt = Checkpoint(name) if name else Checkpoint()
        logger.info(
            f"Gemma4 N300: loading {ckpt.root} for max_batch={max_batch_size} max_seq_len={max_seq_len}"
        )
        model = Gemma4TT(
            mesh_device, ckpt, max_batch=max_batch_size, max_seq_len=max_seq_len
        )
        return cls(model)

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=2, **_):
        """Token slots of the KV pool: KV_POOL_BYTES over the bytes one token slot
        costs on a chip, a K and a V row in each buffer a KV-cache group spans."""
        cfg = Checkpoint(model_name).config if model_name else Checkpoint().config
        per_token = max(
            heads // min(num_devices, heads) * head_dim
            for head_dim, heads in attention_dims(cfg).values()
        )
        layers_per_group = kv_groups(cfg.layer_types).count(0)
        return int(KV_POOL_BYTES / (layers_per_group * 2 * per_token * BFP8_BYTES))

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One spec per layer, so vLLM forms sliding and global KV-cache groups. The
        dtype only sizes vLLM's page accounting; the device stores BFP8."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        text = vllm_config.model_config.hf_config.get_text_config()
        dims = attention_dims(text)
        block = vllm_config.cache_config.block_size
        cls._check_chunk_grid(vllm_config, dims)
        sched = vllm_config.scheduler_config
        if (
            sched.enable_chunked_prefill
        ):  # else whole prompts, in Gemma4TT.prefill's PREFILL_CHUNK pieces
            cls.longest_prefill = min(
                PREFILL_CHUNK,
                -(-sched.max_num_batched_tokens // PREFILL_BUCKET) * PREFILL_BUCKET,
            )
        specs = {}
        for i, kind in enumerate(text.layer_types):
            head_dim, heads = dims[kind]
            common = {
                "block_size": block,
                "num_kv_heads": heads,
                "head_size": head_dim,
                "dtype": torch.bfloat16,
            }
            if kind == "sliding_attention":
                specs[f"model.layers.{i}.self_attn"] = SlidingWindowSpec(
                    **common, sliding_window=text.sliding_window
                )
            else:
                specs[f"model.layers.{i}.self_attn"] = FullAttentionSpec(**common)
        return specs

    @staticmethod
    def _check_chunk_grid(vllm_config, dims):
        """Prefill must start on the KV block grid, the lcm of the group block sizes
        (Gemma4TT.prefill). vLLM's prefix hits fall on it; partial chunks end on the
        plugin's prefill_chunk_alignment, which must therefore be a multiple of it."""
        from vllm_tt_plugin.config import get_tt_prefill_chunk_alignment

        block = vllm_config.cache_config.block_size
        per_token = [heads * head_dim for head_dim, heads in dims.values()]
        grid = math.lcm(*(block * max(per_token) // t for t in per_token))
        if vllm_config.scheduler_config.enable_chunked_prefill:
            align = get_tt_prefill_chunk_alignment(vllm_config)
            if align == 0 or align % grid:
                raise ValueError(
                    f"--block-size {block} puts the KV block grid at {grid} tokens; set "
                    f"additional_config tt.prefill_chunk_alignment to a multiple of it (now {align})"
                )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """per_layer_specs: (shape, dtype, buffer index) per layer from vLLM's plan."""
        m = self.model
        shapes = [tuple(shape) for shape, _, _ in per_layer_specs]
        buffers = [idx for _, _, idx in per_layer_specs]
        m.use_kv(
            KvCache(
                m.mesh,
                m.cfg.layer_types,
                shapes,
                buffers,
                kv_groups(m.cfg.layer_types),
                m.precision.kv_cache,
            )
        )
        # one layer of each group, whose vLLM page table stands for the group's
        self._group_layer = [
            next(i for i, kv in enumerate(m.kv.layers) if kv.group == g)
            for g in range(m.kv.groups)
        ]
        logger.info(
            f"Gemma4 N300: KV pool of {m.kv.num_blocks} blocks, {len(set(buffers))} shared K/V buffers, "
            f"{m.kv.groups} groups with block sizes {m.kv.block_sizes}"
        )
        return m.kv

    def _group_tables(
        self, per_layer: list[torch.Tensor], rows: int, check: bool = False
    ) -> torch.Tensor:
        """vLLM's per-layer page tables as int32 [groups, rows, width]."""
        tables = torch.stack([per_layer[i][:rows] for i in self._group_layer]).to(
            torch.int32
        )
        if check:  # the model's grouping must match vLLM's (see kv_groups)
            for i, kv in enumerate(self.model.kv.layers):
                if not torch.equal(
                    per_layer[i][:rows].to(torch.int32),
                    tables[kv.group, :, : per_layer[i].shape[1]],
                ):
                    raise RuntimeError(
                        f"layer {i} page table differs from its KV-cache group {kv.group}; kv_groups no longer matches vLLM"
                    )
        return tables

    # -- execution -----------------------------------------------------------

    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        page_tables_per_layer=None,
        **_,
    ):
        """Row i computes tokens[i, start_pos[i]:prompt_lens[i]]; the cache holds the positions before."""
        rows = tokens.shape[0]
        tables = self._group_tables(page_tables_per_layer, rows, check=True)
        out = []
        for i in range(rows):
            start, end = int(start_pos[i]), int(prompt_lens[i])
            t0 = time.perf_counter()
            out.append(self.model.prefill(tokens[i, :end], start, tables[:, i]))
            if self.timing:
                logger.info(
                    f"Gemma4 prefill: positions {start}-{end}, {(time.perf_counter() - t0) * 1e3:.0f} ms"
                )
        return torch.stack(out)[:, None, :]

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=False,
        page_tables_per_layer=None,
        sampling_params=None,
        **_,
    ):
        tables = self._group_tables(page_tables_per_layer, tokens.shape[0])
        t0 = time.perf_counter()
        # The model-side sampler caps only what it needs (see sample()).
        raw = sampling_params is not None
        active, logits = self.model.decode_active(
            tokens[:, 0], start_pos, tables, trace=enable_trace, softcap=not raw
        )
        if self.timing:
            t1 = time.perf_counter()
            outside = (
                (t0 - self._last_return) * 1e3 if self._last_return else float("nan")
            )
            n_active = len(active)
            logger.info(
                f"Gemma4 decode: {n_active} active, model {(t1 - t0) * 1e3:.1f} ms, outside since last {outside:.1f} ms"
            )
            self._last_return = t1
        rows = tokens.shape[0]
        if sampling_params is not None:
            ids = torch.zeros(rows, 1, dtype=torch.long)
            log_probs = None
            if len(active):
                got, lp = sample(
                    logits,
                    sampling_params,
                    start_pos[active],
                    rows_idx=active,
                    softcap=self.model.head.softcap,
                )
                ids[active] = got
                if lp is not None:
                    vals = torch.zeros(rows, LOGPROBS_K)
                    lp_ids = torch.zeros(rows, LOGPROBS_K, dtype=torch.long)
                    vals[active], lp_ids[active] = lp
                    log_probs = (vals, lp_ids)
            return ids if log_probs is None else (ids, log_probs)
        full = torch.zeros(rows, logits.shape[-1])
        full[active] = logits
        return full[:, None, :]

    # -- lifecycle -------------------------------------------------------------

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **_):
        """Build the prefill programs of every padded length (prefill_rows), for a piece
        from position 0 and a continuation piece, which reads the paged cache. A program
        built later, after the decode traces are captured, would cost the first request
        of that length and could allocate into trace memory. The pool holds no request
        yet; a small pool (GEMMA4_KV_POOL_GIB) warms only the lengths it fits.

        The plugin calls it again with enable_trace=True; prefill is not traced, so
        the programs from the first call serve."""
        if self._prefill_warmed:
            return
        self._prefill_warmed = True
        m = self.model
        grid = m.kv.grid
        for longest in range(self.longest_prefill, 0, -PREFILL_BUCKET):
            try:
                tables = m.page_tables(1, grid + longest)[:, 0]
                break
            except ValueError:
                continue
        else:
            logger.warning(
                "Gemma4 N300: the KV pool is too small to warm any prefill length"
            )
            return
        t0 = time.perf_counter()
        tokens = torch.zeros(grid + longest, dtype=torch.long)
        for start in (0, grid):
            for n in sorted({prefill_rows(n, start) for n in range(1, longest + 1)}):
                m.prefill(tokens[: start + n], start, tables)
        logger.info(
            f"Gemma4 N300: prefill programs for pieces up to {longest} tokens built in {time.perf_counter() - t0:.0f} s"
        )

    def warmup_model_decode(
        self,
        kv_cache,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        enable_trace,
        **_,
    ):
        """Compile (and capture) one decode program per active row count, since
        decode drops inactive rows."""
        tokens = torch.zeros(max_batch_size, dtype=torch.long)
        tables = self.model.page_tables(max_batch_size, 64)
        for active in range(1, max_batch_size + 1):
            positions = torch.full((max_batch_size,), -1, dtype=torch.long)
            positions[:active] = 0
            self.model.decode(tokens, positions, tables, trace=enable_trace)

    def release_persistent_capture(self):
        self.model.release_traces()
