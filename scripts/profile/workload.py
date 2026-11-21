"""Fixed workload for tt-scope captures: one prefill, then eager decode steps.

    pixi run tt-scope tracy scripts/profile/workload.py -- --prompt-len 128 --decode-steps 2

Signposts mark the phases so the per-op CSV can be split by phase.
"""

import argparse
import time

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.config import Precision
from tt_models.models.gemma4.model import Gemma4TT


def signpost(name):
    try:
        from tracy import signpost as sp

        sp(name)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--precision", default="")
    ap.add_argument("--trace", action="store_true", help="replay a metal trace, so op-to-op gaps are device gaps")
    args = ap.parse_args()

    ckpt = Checkpoint()
    with open_n300(trace_region_size=300_000_000 if args.trace else 0) as mesh:
        model = Gemma4TT(mesh, ckpt, max_batch=args.batch, num_layers=args.layers, precision=Precision.parse(args.precision))
        pts = model.allocate_kv_cache(args.batch, args.prompt_len + args.decode_steps + 2)
        ids = torch.randint(100, 20000, (args.batch, args.prompt_len + args.decode_steps + 2))
        # Warm-up compiles every program once, outside the measured phases.
        for b in range(args.batch):
            model.prefill(ids[b, : args.prompt_len], 0, pts[:, b])
        model.decode(ids[:, args.prompt_len], torch.full((args.batch,), args.prompt_len), pts, trace=args.trace)
        if args.trace:  # the first traced call compiles and captures; replay once more
            model.decode(ids[:, args.prompt_len], torch.full((args.batch,), args.prompt_len), pts, trace=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush, so the device buffers hold only measured phases

        signpost("prefill")
        t0 = time.time()
        model.prefill(ids[0, : args.prompt_len], 0, pts[:, 0])
        ttnn.synchronize_device(mesh)
        print(f"prefill {args.prompt_len}: {time.time() - t0:.3f}s", flush=True)
        ttnn.ReadDeviceProfiler(mesh)
        for step in range(args.decode_steps):
            signpost(f"decode{step}")
            pos = args.prompt_len + 1 + step
            t0 = time.time()
            model.decode(ids[:, pos], torch.full((args.batch,), pos), pts, trace=args.trace)
            ttnn.synchronize_device(mesh)
            print(f"decode step {step}: {time.time() - t0:.3f}s", flush=True)
            ttnn.ReadDeviceProfiler(mesh)
        signpost("end")
        ttnn.ReadDeviceProfiler(mesh)
        model.release_traces()


if __name__ == "__main__":
    main()
