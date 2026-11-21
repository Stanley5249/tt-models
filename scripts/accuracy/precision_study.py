"""Teacher-forced accuracy of several precision configurations over one text.

Every configuration is scored against the float32 HF reference on the same
tokens: next-token perplexity, top-1 agreement, and KL(ref || config).

Configurations:
  fp32          HF reference, float32 compute and BF16 checkpoint weights
  bf16          HF reference in BF16 compute (the stock HF precision)
  emu:<spec>    float32 reference with TTNN weight quantization emulated,
                spec like experts=bfp4,attention=bfp8,mlp=bfp8,lm_head=bfp8
  tt            the device model prefill (all positions)

    pixi run python scripts/accuracy/precision_study.py --tokens 512 --configs fp32,emu:experts=bfp4,tt
"""

import argparse
import json
import time
from pathlib import Path

import torch

from tt_models.models.gemma4.checkpoint import Checkpoint

ROOT = Path(__file__).resolve().parents[2]


def tokens(ckpt, n):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt.root)
    ids = tok((ckpt.root / "README.md").read_text(), return_tensors="pt").input_ids[0]
    return ids[:n]


def parse_quant(spec: str) -> dict:
    import ttnn

    names = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    return {
        k: names[v] for k, v in (item.split("=") for item in spec.split(",") if item)
    }


def run_reference(ckpt, ids, dtype, quant):
    from tt_models.models.gemma4.reference import Reference

    ref = Reference(ckpt, dtype=dtype, quant=quant)
    logits, _ = ref.forward(ids)
    return logits.float()


def run_tt(ckpt, ids, args):
    from tt_models.common.mesh import open_n300
    from tt_models.models.gemma4.model import Gemma4TT

    with open_n300() as mesh:
        model = Gemma4TT(
            mesh, ckpt, cache_root=ROOT / "cache", max_batch=1, precision=args.precision
        )
        tables = model.allocate_kv_cache(1, ids.shape[0])
        return model.prefill(ids, 0, tables[:, 0], all_logits=True)


def score(logits, base, ids):
    n = ids.shape[0] - 1
    lg, bs = logits[:n].float(), base[:n].float()
    nll = torch.nn.functional.cross_entropy(lg, ids[1:])
    logp, logq = torch.log_softmax(bs, -1), torch.log_softmax(lg, -1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    return {
        "ppl": round(nll.exp().item(), 4),
        "top1_vs_fp32": round(
            (lg.argmax(-1) == bs.argmax(-1)).float().mean().item(), 4
        ),
        "kl_mean": round(kl.mean().item(), 5),
        "kl_p95": round(kl.quantile(0.95).item(), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--configs", default="fp32,tt")
    ap.add_argument(
        "--precision",
        default="",
        help='TT Precision overrides, e.g. "expert_fidelity=HiFi2"',
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "accuracy" / "precision_study.json",
    )
    args = ap.parse_args()
    spec = args.precision
    from tt_models.models.gemma4.config import Precision

    args.precision = Precision.parse(spec)

    ckpt = Checkpoint()
    ids = tokens(ckpt, args.tokens)
    base_path = ROOT / "cache" / "ref" / f"readme-{args.tokens}-fp32-logits.pt"
    if base_path.exists():
        base = torch.load(base_path)
    else:
        t0 = time.time()
        base = run_reference(ckpt, ids, torch.float32, {})
        base_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base, base_path)
        print(f"fp32 reference {time.time() - t0:.0f}s", flush=True)

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for config in args.configs.split(";" if ";" in args.configs else ","):
        cfg = config
        t0 = time.time()
        if cfg == "fp32":
            logits = base
        elif cfg == "bf16":
            logits = run_reference(ckpt, ids, torch.bfloat16, {})
        elif cfg.startswith("emu:"):
            logits = run_reference(ckpt, ids, torch.float32, parse_quant(cfg[4:]))
        elif cfg == "tt":
            logits = run_tt(ckpt, ids, args)
            cfg = f"tt[{spec}]" if spec else cfg
        else:
            raise ValueError(cfg)
        results[f"{cfg}@{args.tokens}"] = score(logits, base, ids)
        print(
            cfg,
            results[f"{cfg}@{args.tokens}"],
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del logits
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
