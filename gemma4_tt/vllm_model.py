"""vLLM TT plugin adapter for Gemma 4 26B-A4B on an N300.

Register it for a launch with

    TT_MODEL_CLASS_OVERRIDES=TTGemma4ForConditionalGeneration=gemma4_tt.vllm_model:Gemma4ForCausalLM

vLLM pages only the 5 global-attention layers. The 25 sliding layers keep a
per-slot ring of 1024 tokens that the adapter owns, so the KV spec announced
to vLLM describes one global layer (2 KV heads of 512) without a window.
"""

import torch
import ttnn
from loguru import logger

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.tt.model import Gemma4TT


class Gemma4ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    def __init__(self, model: Gemma4TT):
        self.model = model
        self.model_capabilities = dict(type(self).model_capabilities)

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
            rows.append(self.model.prefill(tokens[i, :end], page_table[i], slot=slot))
        return torch.stack(rows)[:, None, :]

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, enable_trace=False, read_from_device=True, slot_remap=None, **_):
        if slot_remap is not None:
            self.model.remap_slots(slot_remap)
        logits = self.model.decode(tokens[:, 0], start_pos, page_table, trace=enable_trace)
        return logits[:, None, :]

    # -- lifecycle -------------------------------------------------------------

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **_):
        """Compile the prefill programs once on a short dummy prompt."""
        blocks = torch.zeros(self.model.max_seq_len // self.model.block_size, dtype=torch.int32)
        self.model.prefill(torch.zeros(32, dtype=torch.long), blocks, slot=0)

    def warmup_model_decode(self, kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace, **_):
        tokens = torch.zeros(max_batch_size, dtype=torch.long)
        positions = torch.full((max_batch_size,), -1, dtype=torch.long)
        positions[0] = 0
        self.model.decode(tokens, positions, torch.zeros(max_batch_size, num_blocks, dtype=torch.int32), trace=enable_trace)

    def release_request(self, slot):
        pass  # the slot's sliding ring is rewritten by the next prefill into it

    def release_persistent_capture(self):
        self.model.release_traces()
