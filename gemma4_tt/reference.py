"""Layer-streaming CPU reference built from the Hugging Face Gemma 4 modules.

Only one decoder layer lives in host RAM at a time, so the 26B model runs in
float32 within about 6 GB. Use it for teacher-forced logits and per-layer
hidden states; it is slow (tens of seconds per layer for long prompts).
"""

import math

import torch
from transformers.models.gemma4 import modeling_gemma4 as hf

from gemma4_tt.checkpoint import Checkpoint


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


class Reference:
    def __init__(self, ckpt: Checkpoint | None = None, dtype=torch.float32):
        self.ckpt = ckpt or Checkpoint()
        self.dtype = dtype
        config = self.ckpt.config
        config._attn_implementation = "sdpa"
        self.config = config
        self.rotary = hf.Gemma4TextRotaryEmbedding(config)

    def load_layer(self, idx: int) -> hf.Gemma4TextDecoderLayer:
        with torch.device("meta"):
            layer = hf.Gemma4TextDecoderLayer(self.config, idx)
        state = {k: v.to(self.dtype) for k, v in self.ckpt.layer(idx).items()}
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.eval()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        weight = self.ckpt.get("embed_tokens.weight")
        return (weight[ids].float() * embed_scale(self.config.hidden_size)).to(self.dtype)[None]

    @torch.no_grad()
    def layer_forward(self, layer, x: torch.Tensor, positions: torch.Tensor, masks) -> torch.Tensor:
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
        head = self.ckpt.get("embed_tokens.weight").to(self.dtype)
        logits = h[0] @ head.T
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, keep_hidden: bool = False, num_layers: int | None = None):
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
