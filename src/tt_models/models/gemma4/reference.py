"""Layer-streaming CPU reference built from the Hugging Face Gemma 4 modules.

Only one decoder layer lives in host RAM at a time, so the 26B model runs in
float32 within about 6 GB. Use it for teacher-forced logits and per-layer
hidden states; it is slow (tens of seconds per layer for long prompts).
"""

import math

import torch
from transformers.models.gemma4 import modeling_gemma4 as hf

from tt_models.models.gemma4.checkpoint import Checkpoint


def embed_scale(hidden_size: int) -> float:
    # HF casts sqrt(hidden) to the weight dtype; with BF16 weights 53.07 becomes 53.0.
    return torch.tensor(math.sqrt(hidden_size)).to(torch.bfloat16).item()


def attention_masks(seq_len: int, sliding_window: int) -> dict[str, torch.Tensor]:
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    causal = k <= q
    sliding = causal & (k > q - sliding_window)
    neg = torch.finfo(torch.float32).min

    def additive(allowed):
        return torch.where(allowed, 0.0, neg)[None, None]

    return {"full_attention": additive(causal), "sliding_attention": additive(sliding)}


def roundtrip(w: torch.Tensor, ttnn_dtype) -> torch.Tensor:
    """Quantize-dequantize `w` exactly as a TILE upload in `ttnn_dtype` would (host only)."""
    import ttnn

    shape = w.shape
    w4 = w.reshape(-1, shape[-2], shape[-1]) if w.dim() != 4 else w
    t = ttnn.from_torch(
        w4.to(torch.bfloat16), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT
    )
    return ttnn.to_torch(t).to(w.dtype).reshape(shape)


# Device orientation of each quantized weight: blocks of 16 run along the last dim
# of the uploaded [in, out] matrix, so quantize the transpose and transpose back.
QUANT_GROUPS = {
    "attention": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ),
    "mlp": ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"),
    "experts": ("experts.gate_up_proj", "experts.down_proj"),
    "experts_gate_up": ("experts.gate_up_proj",),
    "experts_down": ("experts.down_proj",),
}


def quantize_layer(state: dict, quant: dict) -> dict:
    out = dict(state)
    for group, dtype in quant.items():
        for key in QUANT_GROUPS.get(group, ()):
            if key not in out:
                continue
            w = out[key]
            if key == "experts.gate_up_proj":
                inter = w.shape[1] // 2  # [E, 2I, H] -> device [E, H, I] per half
                halves = [
                    roundtrip(h.transpose(1, 2).contiguous(), dtype).transpose(1, 2)
                    for h in (w[:, :inter], w[:, inter:])
                ]
                out[key] = torch.cat(halves, 1)
            elif key == "experts.down_proj":
                out[key] = (
                    roundtrip(w.transpose(1, 2).contiguous(), dtype)
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                out[key] = roundtrip(w.T.contiguous(), dtype).T.contiguous()
    return out


class Reference:
    def __init__(
        self,
        ckpt: Checkpoint | None = None,
        dtype=torch.float32,
        quant: dict | None = None,
    ):
        """quant: optional {"attention"|"mlp"|"experts"|"lm_head": ttnn dtype} weight emulation."""
        self.ckpt = ckpt or Checkpoint()
        self.dtype = dtype
        self.quant = quant or {}
        config = self.ckpt.config
        config._attn_implementation = "sdpa"  # noqa: SLF001 - Transformers setting
        self.config = config
        self.rotary = hf.Gemma4TextRotaryEmbedding(config)

    def load_layer(self, idx: int) -> hf.Gemma4TextDecoderLayer:
        with torch.device("meta"):
            layer = hf.Gemma4TextDecoderLayer(self.config, idx)
        state = quantize_layer(self.ckpt.layer(idx), self.quant)
        state = {k: v.to(self.dtype) for k, v in state.items()}
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        weight = self.ckpt.get("embed_tokens.weight")
        return (weight[ids].float() * embed_scale(self.config.hidden_size)).to(
            self.dtype
        )[None]

    @torch.no_grad()
    def layer_forward(
        self, layer, x: torch.Tensor, positions: torch.Tensor, masks
    ) -> torch.Tensor:
        layer_type = self.config.layer_types[layer.layer_idx]
        cos_sin = self.rotary(x, positions[None], layer_type)
        return layer(
            x,
            shared_kv_states={},
            position_embeddings=cos_sin,
            attention_mask=masks[layer_type],
            position_ids=positions[None],
        )

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        norm = hf.Gemma4RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight.data = self.ckpt.get("norm.weight").to(self.dtype)
        h = norm(x)
        head = self.ckpt.get("embed_tokens.weight")
        if "lm_head" in self.quant:
            head = roundtrip(head.T.contiguous(), self.quant["lm_head"]).T
        head = head.to(self.dtype)
        logits = h[0] @ head.T
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    @torch.no_grad()
    def forward(
        self,
        ids: torch.Tensor,
        keep_hidden: bool = False,
        num_layers: int | None = None,
    ):
        """Teacher-forced forward over `ids` (1D). Returns logits [S, V] and optional hiddens."""
        positions = torch.arange(ids.shape[0])
        masks = attention_masks(ids.shape[0], self.config.sliding_window)
        x = self.embed(ids)
        hiddens = [x[0].clone()] if keep_hidden else None
        for idx in range(num_layers or self.config.num_hidden_layers):
            layer = self.load_layer(idx)
            x = self.layer_forward(layer, x, positions, masks)
            del layer
            if keep_hidden:
                hiddens.append(x[0].clone())
        return self.logits(x), hiddens
