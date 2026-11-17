"""A random-weight stand-in for Checkpoint with the real Gemma 4 26B-A4B shapes."""

import copy

import torch
from transformers import AutoConfig

from gemma4_tt.checkpoint import resolve_snapshot


class FakeCheckpoint:
    def __init__(self, layer_types: list[str] | None = None, seed: int = 0):
        cfg = copy.deepcopy(AutoConfig.from_pretrained(resolve_snapshot()).get_text_config())
        if layer_types:
            cfg.layer_types = list(layer_types) + cfg.layer_types[len(layer_types) :]
        self.config = cfg
        self.seed = seed
        self._cache = {}

    def _shape(self, name: str):
        c = self.config
        H, E = c.hidden_size, c.num_experts
        if name in ("embed_tokens.weight",):
            return (c.vocab_size, H), 1.0
        if name == "norm.weight":
            return (H,), 0.1
        idx = int(name.split(".")[1])
        key = name.split(".", 2)[2]
        sliding = c.layer_types[idx] == "sliding_attention"
        hd = c.head_dim if sliding else c.global_head_dim
        nkv = c.num_key_value_heads if sliding else c.num_global_key_value_heads
        nq = c.num_attention_heads
        table = {
            "self_attn.q_proj.weight": ((nq * hd, H), H**-0.5),
            "self_attn.k_proj.weight": ((nkv * hd, H), H**-0.5),
            "self_attn.v_proj.weight": ((nkv * hd, H), H**-0.5),
            "self_attn.o_proj.weight": ((H, nq * hd), (nq * hd) ** -0.5),
            "self_attn.q_norm.weight": ((hd,), 0.1),
            "self_attn.k_norm.weight": ((hd,), 0.1),
            "mlp.gate_proj.weight": ((c.intermediate_size, H), H**-0.5),
            "mlp.up_proj.weight": ((c.intermediate_size, H), H**-0.5),
            "mlp.down_proj.weight": ((H, c.intermediate_size), c.intermediate_size**-0.5),
            "router.proj.weight": ((E, H), 1.0),
            "router.scale": ((H,), 0.1),
            "router.per_expert_scale": ((E,), 0.1),
            "experts.gate_up_proj": ((E, 2 * c.moe_intermediate_size, H), H**-0.5),
            "experts.down_proj": ((E, H, c.moe_intermediate_size), c.moe_intermediate_size**-0.5),
            "layer_scalar": ((1,), 0.0),
        }
        if key.endswith("layernorm.weight") or "layernorm_" in key:
            return (H,), 0.1
        return table[key]

    def get(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            shape, std = self._shape(name)
            g = torch.Generator().manual_seed(hash((self.seed, name)) % (2**31))
            t = torch.randn(shape, generator=g) * std
            if name.endswith("norm.weight") or name.endswith("layernorm.weight") or "layernorm_" in name or name.endswith("scale"):
                t = 1.0 + t
            if name.endswith("layer_scalar"):
                t = torch.ones(1)
            t = t.to(torch.bfloat16)
            if t.numel() > 10_000_000:
                return t  # do not keep the large ones
            self._cache[name] = t
        return self._cache[name]
