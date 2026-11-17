"""Fixed workload for tt-scope captures: one prefill, then eager decode steps.

    pixi run tt-scope tracy scripts/profile_workload.py -- --prompt-len 128 --decode-steps 2

Signposts mark the phases so the per-op CSV can be split by phase.
"""

import argparse
import time

import torch
import ttnn

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.mesh import open_n300
from gemma4_tt.tt.model import Gemma4TT, Precision


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
    args = ap.parse_args()

    ckpt = Checkpoint()
    with open_n300() as mesh:
        model = Gemma4TT(mesh, ckpt, max_batch=args.batch, num_layers=args.layers, precision=Precision.parse(args.precision))
        per_user = -(-(args.prompt_len + args.decode_steps + 128) // model.block_size)
        model.allocate_kv_cache(per_user * args.batch + 1)
        pts = torch.arange(per_user * args.batch, dtype=torch.int32).reshape(args.batch, per_user)
        ids = torch.randint(100, 20000, (args.batch, args.prompt_len + args.decode_steps + 2))
        # Warm-up compiles every program once, outside the measured phases.
        for b in range(args.batch):
            model.prefill(ids[b, : args.prompt_len], pts[b], slot=b)
        model.decode(ids[:, args.prompt_len], torch.full((args.batch,), args.prompt_len), pts)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush, so the device buffers hold only measured phases

        signpost("prefill")
        t0 = time.time()
        model.prefill(ids[0, : args.prompt_len], pts[0], slot=0)
        ttnn.synchronize_device(mesh)
        print(f"prefill {args.prompt_len}: {time.time() - t0:.3f}s", flush=True)
        ttnn.ReadDeviceProfiler(mesh)
        for step in range(args.decode_steps):
            signpost(f"decode{step}")
            pos = args.prompt_len + 1 + step
            t0 = time.time()
            model.decode(ids[:, pos], torch.full((args.batch,), pos), pts)
            ttnn.synchronize_device(mesh)
            print(f"decode step {step}: {time.time() - t0:.3f}s", flush=True)
            ttnn.ReadDeviceProfiler(mesh)
        signpost("end")
        ttnn.ReadDeviceProfiler(mesh)


if __name__ == "__main__":
    main()
