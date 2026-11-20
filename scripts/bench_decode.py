"""Model decode latency on the device, outside vLLM.

    pixi run python scripts/bench_decode.py --batch 1 --steps 32 [--trace] [--out results/decode/x.json]

Prefills every slot with the same prompt, then times `--steps` decode calls
end to end (host inputs in, float logits out). The first call per mode is a
warm-up (compile, or trace capture) and is excluded.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.mesh import open_n300
from gemma4_tt.tt.model import Gemma4TT, Precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--precision", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = Checkpoint()
    with open_n300(trace_region_size=300_000_000 if args.trace else 0) as mesh:
        t0 = time.time()
        model = Gemma4TT(mesh, ckpt, max_batch=args.batch, num_layers=args.layers, precision=Precision.parse(args.precision))
        print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
        pts = model.allocate_kv_cache(args.batch, args.prompt_len + args.steps + 1)
        torch.manual_seed(0)
        ids = torch.randint(100, 20000, (args.prompt_len,))
        for b in range(args.batch):
            logits = model.prefill(ids, 0, pts[:, b])
        tok = torch.full((args.batch,), int(logits.argmax()))
        times = []
        for step in range(args.steps + 1):
            pos = torch.full((args.batch,), args.prompt_len + step)
            t0 = time.perf_counter()
            logits = model.decode(tok, pos, pts, trace=args.trace)
            dt = time.perf_counter() - t0
            tok = logits.argmax(-1)
            if step:
                times.append(dt * 1e3)
        ttnn.synchronize_device(mesh)
        model.release_traces()
    med = statistics.median(times)
    res = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "median_ms": round(med, 2),
        "mean_ms": round(statistics.mean(times), 2),
        "min_ms": round(min(times), 2),
        "tok_s_per_user": round(1e3 / med, 2),
        "tok_s_total": round(args.batch * 1e3 / med, 2),
    }
    print(json.dumps(res))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res | {"times_ms": [round(t, 2) for t in times]}, indent=1))


if __name__ == "__main__":
    main()
