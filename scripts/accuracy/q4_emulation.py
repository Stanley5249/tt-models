"""Teacher-forced accuracy of emulated expert weight formats on the CPU reference.

    pixi run python scripts/accuracy/q4_emulation.py \\
        artifacts/q4/emulation.json q4exact q4clamp bfp4 [...]
    EMU_TOKENS=512 EMU_DTYPE=bfloat16 pixi run python \\
        scripts/accuracy/q4_emulation.py out.json none q4clamp

Every routed-expert weight of the HF reference is replaced by one format, and the
README logits are scored against the cached fp32 reference (precision_study.py):
- none: the checkpoint weights (with EMU_DTYPE=bfloat16, BF16 compute alone);
- q4exact: q * s from the factored Q4_0 integers and steps (the BF16 checkpoint
  is q * s rounded to BF16);
- q4clamp: as q4exact with q = -8 clamped to -7 (the device Q4 kernels);
- q4shift: q4exact for blocks whose integers fit -7..7 after adding 1, clamped
  otherwise (the blocks a +1 offset could store exactly);
- bfp4: the device's BFP4 tiles (blocks of 16 along the outputs).
EMU_DENSE=bfp8|q4clamp|q4exact also replaces the attention, dense MLP and LM-head
weights (the device stores them as BFP8 today).
No ttnn import: a ttnn host conversion starts the device cluster, which breaks a
device job running next to this one.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from precision_study import score, tokens  # a sibling script

from tt_models.models.gemma4 import reference
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.q4 import factor_blocks

ROOT = Path(__file__).resolve().parents[2]


def q4(w: torch.Tensor, mode: str) -> torch.Tensor:
    if w.shape[0] > 8:  # bounds host memory
        return torch.cat([q4(w[i : i + 8], mode) for i in range(0, w.shape[0], 8)])
    b = w.float().reshape(*w.shape[:-1], -1, 32)
    q, s = factor_blocks(b)
    if mode == "q4clamp":
        q = q.clamp(-7, 7)
    elif mode == "q4shift":
        shift = (q.amin(-1, keepdim=True) == -8) & (q.amax(-1, keepdim=True) <= 6)
        q = torch.where(shift, q, q.clamp(-7, 7))
    return (q * s).reshape(w.shape).to(w.dtype)


def bfp4(w: torch.Tensor) -> torch.Tensor:
    """Bit-exact torch BFP4 tile rounding, blocks of 16 along the last dim."""
    b = w.to(torch.bfloat16).float().reshape(-1, 16)
    e = torch.floor(torch.log2(b.abs().amax(-1, keepdim=True).clamp(min=1e-38)))
    st = torch.exp2(e - 2)
    return ((b / st).round().clamp(-7, 7) * st).reshape(w.shape).to(w.dtype)


def bfp8(w: torch.Tensor) -> torch.Tensor:
    """Torch BFP8 tile rounding: blocks of 16 along the last dim, sign + 7-bit
    magnitude.
    """
    b = w.to(torch.bfloat16).float().reshape(-1, 16)
    e = torch.floor(torch.log2(b.abs().amax(-1, keepdim=True).clamp(min=1e-38)))
    st = torch.exp2(e - 6)
    return ((b / st).round().clamp(-127, 127) * st).reshape(w.shape).to(w.dtype)


def dense(w: torch.Tensor, mode: str) -> torch.Tensor:
    """An HF [out, in] weight in `mode`; the device stores [in, out], so BFP blocks run
    along the outputs.
    """
    if mode == "bfp8":
        return bfp8(w.T.contiguous()).T.contiguous()
    if w.shape[0] > 4096:
        return torch.cat(
            [dense(w[i : i + 4096], mode) for i in range(0, w.shape[0], 4096)]
        )
    return q4(w[None], mode)[0]


DENSE_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def main():
    out = Path(sys.argv[1])
    modes = sys.argv[2:]
    state = {"mode": None}

    def quantize_layer(layer_state, _quant):
        result = dict(layer_state)
        if state["mode"] == "none":
            return result
        for key in (
            "experts.gate_up_proj",
            "experts.down_proj",
        ):  # HF layout [E, out, in]
            w = result[key]
            if (
                state["mode"] == "bfp4"
            ):  # the device stores [in, out]: blocks along the outputs
                result[key] = (
                    bfp4(w.transpose(-1, -2).contiguous())
                    .transpose(-1, -2)
                    .contiguous()
                )
            else:
                result[key] = q4(w, state["mode"])
        return result

    dense_mode = os.environ.get("EMU_DENSE", "")
    plain_quantize = quantize_layer

    def quantize_all(layer_state, quant):
        result = plain_quantize(layer_state, quant)
        if dense_mode:
            for key in DENSE_KEYS:
                if key in result:
                    result[key] = dense(result[key], dense_mode)
        return result

    reference.quantize_layer = quantize_all
    # Reference.logits sends the LM head [H, V] through roundtrip when "lm_head" is
    # quantized
    if dense_mode:
        reference.roundtrip = lambda head_t, _dtype: dense(
            head_t.T.contiguous(), dense_mode
        ).T.contiguous()
    assert "ttnn" not in sys.modules, "ttnn would start the device cluster"
    ck = Checkpoint()
    n = int(os.environ.get("EMU_TOKENS", "2048"))
    dtype = getattr(torch, os.environ.get("EMU_DTYPE", "float32"))
    ids = tokens(ck, n)
    base = torch.load(ROOT / "cache" / "ref" / f"readme-{n}-fp32-logits.pt")
    results = json.loads(out.read_text()) if out.exists() else {}
    for mode in modes:
        state["mode"] = mode
        t0 = time.time()
        logits, _ = reference.Reference(
            ck, dtype=dtype, quant={"lm_head": "emulated"} if dense_mode else None
        ).forward(ids)
        key = f"{mode}{'+dense-' + dense_mode if dense_mode else ''}@{n}" + (
            "" if dtype == torch.float32 else f"-{str(dtype).removeprefix('torch.')}"
        )
        results[key] = score(logits.float(), base, ids)
        print(key, results[key], f"{time.time() - t0:.0f}s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
