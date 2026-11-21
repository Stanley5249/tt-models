"""A random-weight stand-in for Checkpoint with the real Gemma 4 26B-A4B shapes."""

import zlib

import torch
from transformers import AutoConfig

from tt_models.models.gemma4.checkpoint import attention_dims, resolve_snapshot


class FakeCheckpoint:
    def __init__(self, layer_types: list[str] | None = None, seed: int = 0):
        cfg = AutoConfig.from_pretrained(resolve_snapshot()).get_text_config()
        if layer_types:
            # per_layer_config holds the global layers' head_dim and KV heads by
            # index, so it is rebuilt to follow the new layer types.
            d = cfg.to_dict()
            d["layer_types"] = list(layer_types) + cfg.layer_types[len(layer_types) :]
            full = next(
                cfg.per_layer_config[i]
                for i, t in enumerate(cfg.layer_types)
                if t == "full_attention"
            )
            values = {
                "head_dim": full.head_dim,
                "num_key_value_heads": full.num_key_value_heads,
            }
            d["per_layer_config"] = {
                f"{i:02d}": values
                for i, t in enumerate(d["layer_types"])
                if t == "full_attention"
            }
            cfg = type(cfg)(**d)
        self.config = cfg
        self.seed = seed
        self._cache = {}

    def _shape(self, name: str):
        c = self.config
        H, E = c.hidden_size, c.num_experts
        if name == "embed_tokens.weight":
            return (c.vocab_size, H), 1.0
        if name == "norm.weight":
            return (H,), 0.1
        idx = int(name.split(".")[1])
        key = name.split(".", 2)[2]
        hd, nkv = attention_dims(c)[c.layer_types[idx]]
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
            "mlp.down_proj.weight": (
                (H, c.intermediate_size),
                c.intermediate_size**-0.5,
            ),
            "router.proj.weight": ((E, H), 1.0),
            "router.scale": ((H,), 0.1),
            "router.per_expert_scale": ((E,), 0.1),
            "experts.gate_up_proj": ((E, 2 * c.moe_intermediate_size, H), H**-0.5),
            "experts.down_proj": (
                (E, H, c.moe_intermediate_size),
                c.moe_intermediate_size**-0.5,
            ),
            "layer_scalar": ((1,), 0.0),
        }
        if key.endswith("layernorm.weight") or "layernorm_" in key:
            return (H,), 0.1
        return table[key]

    def get(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            shape, std = self._shape(name)
            # crc32, not hash(): str hashes change per process, and tensors read without the
            # weight cache must match the cached ones a previous process built.
            g = torch.Generator().manual_seed(
                zlib.crc32(f"{self.seed}:{name}".encode())
            )
            t = torch.randn(shape, generator=g) * std
            if (
                name.endswith(("norm.weight", "layernorm.weight", "scale"))
                or "layernorm_" in name
            ):
                t = 1.0 + t
            if name.endswith("layer_scalar"):
                t = torch.ones(1)
            t = t.to(torch.bfloat16)
            if t.numel() > 10_000_000:
                return t  # do not keep the large ones
            self._cache[name] = t
        return self._cache[name]
