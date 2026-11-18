"""vLLM TT plugin adapter for Gemma 4 26B-A4B on an N300.

Register it for a launch with

    TT_MODEL_CLASS_OVERRIDES=TTGemma4ForConditionalGeneration=gemma4_tt.vllm_model:Gemma4ForCausalLM

vLLM pages only the 5 global-attention layers. The 25 sliding layers keep a
per-slot ring of 1024 tokens that the adapter owns, so the KV spec announced
to vLLM describes one global layer (2 KV heads of 512) without a window.
"""

import os
import time

import torch
from loguru import logger

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.tt.model import Gemma4TT

# Largest top-k the model-side sampler takes; requests with no top-k (or a
# larger one) and a temperature fall back to vLLM's host sampler.
MAX_TOP_K = 1024
LOGPROBS_K = 32


def sample(logits: torch.Tensor, params, positions: torch.Tensor, rows_idx=None):
    """Sample one token per row from float logits [B, V] with vLLM's semantics.

    rows_idx: the batch rows these logits belong to (their sampling params); default 0..B-1.

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
    if all(t == 0.0 for t in temps) and (params.enable_log_probs is None or not any(params.enable_log_probs[r] for r in idx)):
        return logits.argmax(-1, keepdim=True), None  # greedy: no top-k sort
    top_ks = [int(params.top_k[r]) if int(params.top_k[r]) > 0 else MAX_TOP_K for r in idx]
    k_max = min(max(top_ks + [1]), MAX_TOP_K)
    vals, ids = logits.topk(k_max, dim=-1)  # sorted, largest first
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
        gen = None if seed is None else torch.Generator().manual_seed((int(seed) * 1_000_003 + int(positions[i])) % 2**63)
        out[i, 0] = ids[i, torch.multinomial(probs, 1, generator=gen)]
    log_probs = None
    if params.enable_log_probs is not None and any(params.enable_log_probs[r] for r in idx):
        lp, lp_ids = torch.log_softmax(logits, dim=-1).topk(LOGPROBS_K, dim=-1)
        log_probs = (lp, lp_ids)
    return out, log_probs


class Gemma4ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_async_decode": False,
        # "Device" sampling here is the host-side top-k sampler above, which the
        # plugin calls through decode_forward(sampling_params=...).
        "supports_sample_on_device": True,
        "max_device_top_k": MAX_TOP_K,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
    }

    def __init__(self, model: Gemma4TT):
        self.model = model
        self.model_capabilities = dict(type(self).model_capabilities)
        # GEMMA4_TIMING=1 logs the model's share of every step, to separate it from vLLM's.
        self.timing = os.environ.get("GEMMA4_TIMING") == "1"
        self._last_return = None

    # -- setup ---------------------------------------------------------------

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **_):
        if tt_data_parallel != 1:
            raise ValueError("Gemma 4 26B-A4B on N300 uses one 1x2 mesh; data parallelism is not supported")
        name = getattr(hf_config, "name_or_path", None) or getattr(hf_config, "_name_or_path", None)
        ckpt = Checkpoint(name) if name else Checkpoint()
        logger.info(f"Gemma4 N300: loading {ckpt.root} for max_batch={max_batch_size} max_seq_len={max_seq_len}")
        model = Gemma4TT(mesh_device, ckpt, max_batch=max_batch_size, max_seq_len=max_seq_len)
        return cls(model)

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=2, tt_data_parallel=1, max_model_len=32768, max_num_seqs=4, **_):
        return max_model_len * max_num_seqs

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        text = vllm_config.model_config.hf_config.get_text_config()
        spec = FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=text.num_global_key_value_heads,
            head_size=text.global_head_dim,
            dtype=torch.bfloat16,
        )
        return {"foo": spec}

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        num_blocks, _, block_size, _ = kv_cache_shape
        self.model.block_size = block_size
        logger.info(f"Gemma4 N300: KV pool of {num_blocks} blocks x {block_size} tokens for the global layers")
        return self.model.allocate_kv_cache(num_blocks)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        shape, dtype, _ = per_layer_specs[0]
        return self.allocate_kv_cache(shape, dtype, len(per_layer_specs))

    # -- execution -----------------------------------------------------------

    def prefill_forward(self, tokens, start_pos, prompt_lens, page_table, kv_cache, enable_trace=False, empty_slots=None, **_):
        rows = []
        for i in range(tokens.shape[0]):
            start, end = int(start_pos[i]), int(prompt_lens[i])
            if start != 0:
                raise NotImplementedError(f"prefill must start at position 0, got {start}")
            slot = int(empty_slots[i]) if empty_slots is not None else i
            t0 = time.perf_counter()
            rows.append(self.model.prefill(tokens[i, :end], page_table[i], slot=slot))
            if self.timing:
                logger.info(f"Gemma4 prefill: {end} tokens in slot {slot}, {(time.perf_counter() - t0) * 1e3:.0f} ms")
        return torch.stack(rows)[:, None, :]

    def decode_forward(
        self, tokens, start_pos, page_table, kv_cache, enable_trace=False, read_from_device=True, slot_remap=None, sampling_params=None, **_
    ):
        if slot_remap is not None:
            self.model.remap_slots(slot_remap)
        t0 = time.perf_counter()
        active, logits = self.model.decode_active(tokens[:, 0], start_pos, page_table, trace=enable_trace)
        if self.timing:
            t1 = time.perf_counter()
            outside = (t0 - self._last_return) * 1e3 if self._last_return else float("nan")
            n_active = len(active)
            logger.info(f"Gemma4 decode: {n_active} active, model {(t1 - t0) * 1e3:.1f} ms, outside since last {outside:.1f} ms")
            self._last_return = t1
        rows = tokens.shape[0]
        if sampling_params is not None:
            ids = torch.zeros(rows, 1, dtype=torch.long)
            log_probs = None
            if len(active):
                got, lp = sample(logits, sampling_params, start_pos[active], rows_idx=active)
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
        """Compile the prefill programs once on a short dummy prompt."""
        blocks = torch.zeros(self.model.max_seq_len // self.model.block_size, dtype=torch.int32)
        self.model.prefill(torch.zeros(32, dtype=torch.long), blocks, slot=0)

    def warmup_model_decode(self, kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace, **_):
        """Compile (and capture) one decode program per active row count, since
        decode drops inactive rows; position 0 of slot 0 is rewritten by any prefill."""
        tokens = torch.zeros(max_batch_size, dtype=torch.long)
        page_table = torch.zeros(max_batch_size, num_blocks, dtype=torch.int32)
        for active in range(1, max_batch_size + 1):
            positions = torch.full((max_batch_size,), -1, dtype=torch.long)
            positions[:active] = 0
            self.model.decode(tokens, positions, page_table, trace=enable_trace)

    def release_request(self, slot):
        pass  # the slot's sliding ring is rewritten by the next prefill into it

    def release_persistent_capture(self):
        self.model.release_traces()
