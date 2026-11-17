"""Teacher-forced comparison of the TT model against the HF CPU reference.

The TT side prefills `--prompt-len` tokens and then decodes the following
`--decode-steps` reference tokens one at a time, so every compared row has the
same context on both sides. Reference logits are cached under cache/ref/.
With several `--offsets`, each one starts a text segment at that token of the
README and the rows of all segments are pooled (one model load).

    pixi run python scripts/compare.py --layers 6 --prompt-len 128 --decode-steps 8
    pixi run python scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gemma4_tt.checkpoint import Checkpoint

ROOT = Path(__file__).resolve().parents[1]


def prompt_tokens(ckpt: Checkpoint, n: int) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained(ckpt.root)
    text = (ckpt.root / "README.md").read_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    while ids.shape[0] < n:
        ids = torch.cat([ids, ids])
    return ids[:n]


def reference_logits(ckpt, ids, first, layers, offset=0) -> torch.Tensor:
    tag = f"-at{offset}" if offset else ""
    path = ROOT / "cache" / "ref" / f"readme{tag}-n{ids.shape[0]}-from{first}-L{layers}.pt"
    if path.exists():
        return torch.load(path)
    from gemma4_tt.reference import Reference

    t0 = time.time()
    ref = Reference(ckpt)
    positions = torch.arange(ids.shape[0])
    from gemma4_tt.reference import attention_masks

    masks = attention_masks(ids.shape[0], ref.config.sliding_window)
    x = ref.embed(ids)
    for i in range(layers):
        layer = ref.load_layer(i)
        x = ref.layer_forward(layer, x, positions, masks)
        del layer
        print(f"reference layer {i} done ({time.time() - t0:.0f}s)", flush=True)
    logits = ref.logits(x[:, first:])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, path)
    return logits


def metrics(tt: torch.Tensor, ref: torch.Tensor) -> dict:
    tt, ref = tt.float(), ref.float()
    pcc = torch.corrcoef(torch.stack([tt, ref]))[0, 1].item()
    t5, r5 = tt.topk(5).indices, ref.topk(5).indices
    p_ref = torch.softmax(ref, -1)
    kl = torch.sum(p_ref * (torch.log_softmax(ref, -1) - torch.log_softmax(tt, -1))).item()
    return {
        "pcc": round(pcc, 5),
        "top1": bool(t5[0] == r5[0]),
        "top5_overlap": len(set(t5.tolist()) & set(r5.tolist())),
        "kl": round(kl, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    ckpt = Checkpoint()
    total = args.prompt_len + args.decode_steps
    text = prompt_tokens(ckpt, max(args.offsets) + total + 1)
    segments = []
    for offset in args.offsets:
        ids = text[offset : offset + total + 1]
        segments.append((offset, ids, reference_logits(ckpt, ids[:total], args.prompt_len - 1, args.layers, offset)))

    import ttnn

    from gemma4_tt.mesh import open_n300
    from gemma4_tt.tt.model import Gemma4TT

    rows = []
    with open_n300(trace_region_size=200_000_000 if args.trace else 0) as mesh:
        t0 = time.time()
        model = Gemma4TT(mesh, ckpt, cache_root=ROOT / "cache", max_batch=1, num_layers=args.layers)
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
        blocks = -(-(total + 64) // model.block_size)
        model.allocate_kv_cache(blocks)
        pt = torch.arange(blocks, dtype=torch.int32)
        for offset, ids, ref in segments:
            t0 = time.time()
            logits = model.prefill(ids[: args.prompt_len], pt, slot=0)
            print(f"prefill {args.prompt_len} tokens at {offset}: {time.time() - t0:.2f}s", flush=True)
            rows.append({"offset": offset, "pos": args.prompt_len - 1, **metrics(logits, ref[0])})
            for step in range(args.decode_steps):
                pos = args.prompt_len + step
                t0 = time.time()
                logits = model.decode(ids[pos : pos + 1], torch.tensor([pos]), pt[None], trace=args.trace)
                dt = time.time() - t0
                rows.append({"offset": offset, "pos": pos, "sec": round(dt, 3), **metrics(logits[0], ref[step + 1])})
        ttnn.synchronize_device(mesh)
    for r in rows:
        print(json.dumps(r))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"args": vars(args) | {"out": str(args.out)}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
