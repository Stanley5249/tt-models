"""One Gemma 4 decoder layer: attention and feed-forward with their norms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from tt_models.common.ccl import all_gather_compact
from tt_models.common.norm import rms_norm
from tt_models.common.weights import WeightLoader, norm_weight
from tt_models.models.gemma4.attention import Attention, CompactRope, PrefillChunk
from tt_models.models.gemma4.compact import compact_tensor
from tt_models.models.gemma4.moe import FeedForward

if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from tt_models.models.gemma4.boundary import Boundary
    from tt_models.models.gemma4.checkpoint import WeightSource
    from tt_models.models.gemma4.config import Precision
    from tt_models.models.gemma4.kv_cache import LayerKv


class DecoderLayer:
    """Combine attention and feedforward for one decoder layer."""

    def __init__(
        self,
        loader: WeightLoader,
        ckpt: WeightSource,
        cfg: Gemma4TextConfig,
        idx: int,
        precision: Precision,
    ) -> None:
        """Load the weights and operators for one layer."""
        p = f"layers.{idx}."
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.idx = idx
        assert cfg.layer_types is not None
        self.sliding = cfg.layer_types[idx] == "sliding_attention"
        self.eps = cfg.rms_norm_eps
        self.attn = Attention(loader, ckpt, cfg, idx, precision.attention)
        self.ff = FeedForward(
            loader,
            ckpt,
            cfg,
            idx,
            precision.mlp,
            precision.experts,
            precision.expert_fidelity,
            expert_q4=precision.expert_q4,
        )
        self.input_ln = loader.put(
            f"L{idx}.input_ln",
            lambda: norm_weight(ckpt.get(p + "input_layernorm.weight")),
            layout=rm,
        )
        self.post_attn_ln = loader.put(
            f"L{idx}.post_attn_ln",
            lambda: norm_weight(ckpt.get(p + "post_attention_layernorm.weight")),
            layout=rm,
        )
        self.scalar = float(ckpt.get(p + "layer_scalar").float().item())
        self._raw = {
            n: ckpt.get(p + n + ".weight").float()
            for n in (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "post_feedforward_layernorm",
                "input_layernorm",
            )
        }

    @property
    def input_norm(self) -> torch.Tensor:
        """Return the host input normalization weight."""
        return self._raw["input_layernorm"]

    def boundary_weights(self, mesh: ttnn.MeshDevice, next_norm: torch.Tensor) -> None:
        """Load the compact weights for the two batch-1 tails.

        Compact FP32 weights of the two batch-1 tails; next_norm: the next layer's
        input norm (or the final norm).
        """
        r = self._raw
        attn = torch.stack(
            [
                r["post_attention_layernorm"],
                r["pre_feedforward_layernorm"],
                r["pre_feedforward_layernorm_2"],
            ]
        )
        ff = torch.stack(
            [
                r["post_feedforward_layernorm_1"],
                r["post_feedforward_layernorm_2"],
                r["post_feedforward_layernorm"],
                next_norm,
            ]
        )
        self.w_attn_c = compact_tensor(mesh, attn)
        self.w_ff_c = compact_tensor(mesh, ff)

    def _rms(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        return rms_norm(x, w, self.eps)

    def _finish(self, x: ttnn.Tensor, attn_out: ttnn.Tensor) -> ttnn.Tensor:
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

    def prefill(
        self,
        x: ttnn.Tensor,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        kv: LayerKv,
        chunk: PrefillChunk,
    ) -> ttnn.Tensor:
        """Run one decoder layer over a prefill chunk."""
        h = self._rms(x, self.input_ln)
        a = self.attn.prefill(h, rope, kv, chunk)
        h.deallocate(True)
        return self._finish(x, a)

    def decode_compact(
        self,
        xc: ttnn.Tensor,
        h: ttnn.Tensor,
        rope: CompactRope,
        kv: LayerKv,
        page_table: ttnn.Tensor,
        cur_pos: ttnn.Tensor,
        b: Boundary,
        x_page0: int = 0,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Decode one layer using compact boundaries.

        1-4 decode tokens on compact boundaries: xc compact FP32 residuals (one
        vector per row), h their
        normed row form [1, 1, rows, H]. Returns the next (xc, h); h of the last layer
        is final-normed.
        """
        rows = h.shape[2]
        a = self.attn.decode_compact(h, rope, kv, page_table, cur_pos, b)
        g = all_gather_compact(a)
        a.deallocate(True)
        x2, n, m_in, e_in = b.compact(rows), b.row(rows), b.row(rows), b.row(rows)
        b.attn_tail(g, xc, self.w_attn_c, x2, n, m_in, e_in, x_page0=x_page0, rows=rows)
        g.deallocate(True)
        f = self.ff.decode_compact(n, m_in, e_in, b)
        for t in (n, m_in, e_in):
            t.deallocate(True)
        g = all_gather_compact(f)
        f.deallocate(True)
        x3, h2 = b.compact(rows), b.row(rows)
        b.ff_tail(g, x2, self.w_ff_c, x3, h2, self.scalar, rows=rows)
        g.deallocate(True)
        x2.deallocate(True)
        return x3, h2
