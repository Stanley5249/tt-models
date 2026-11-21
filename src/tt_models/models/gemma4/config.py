"""Gemma 4 model settings: weight precisions, prefill pieces, decode rows."""

from dataclasses import dataclass

import ttnn

from tt_models.common.config import TILE

# Decode batches up to this many tokens take the expert streams (their scratch holds 32
# slots).
STREAM_ROWS = 4
# Longest piece of a prompt one prefill pass computes. It bounds activation memory
# (the expert-major MoE gathers k = 8 rows per token) whatever chunk vLLM schedules.
PREFILL_CHUNK = 4096
# Prefill pieces are padded to a multiple of this many rows, so their programs are
# built once per bucket (warmed at startup) instead of once per prompt length: a
# never-seen length cost 0.6 s more on its first prefill. The padding costs about
# 0.5 ms per row, so a short piece from position 0 (up to SHORT_PIECE tokens) pads
# only to whole tiles; a later piece needs whole 128-row SDPA chunks anyway.
PREFILL_BUCKET = 128
SHORT_PIECE = 512


def prefill_rows(tokens: int, start: int) -> int:
    """The rows a prefill piece of `tokens` positions from `start` is padded to."""
    step = TILE if start == 0 and tokens <= SHORT_PIECE else PREFILL_BUCKET
    return -(-tokens // step) * step


@dataclass
class Precision:
    attention: ttnn.DataType = ttnn.bfloat8_b
    mlp: ttnn.DataType = ttnn.bfloat8_b
    experts: ttnn.DataType = ttnn.bfloat4_b
    lm_head: ttnn.DataType = ttnn.bfloat8_b
    kv_cache: ttnn.DataType = ttnn.bfloat8_b
    expert_fidelity: str = "LoFi"  # compute fidelity of the expert sparse matmuls
    expert_q4: bool = (
        False  # experimental Q4_0 experts (expert_q4.py): batch-1 decode only
    )

    @classmethod
    def parse(cls, spec: str) -> "Precision":
        """From "experts=bfp4,expert_fidelity=HiFi2"-style overrides."""
        names = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        kwargs = {}
        for item in filter(None, spec.split(",")):
            key, value = item.split("=")
            kwargs[key] = (
                value
                if key.endswith("fidelity")
                else value == "1"
                if key == "expert_q4"
                else names[value]
            )
        return cls(**kwargs)

    def tag(self) -> str:
        names = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}
        return "-".join(
            names[d] for d in (self.attention, self.mlp, self.experts, self.lm_head)
        )
