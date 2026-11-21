"""Soak test: many traced decode steps, checking device DRAM use does not grow.

    pixi run python scripts/bench/soak.py --steps 3000
"""

import argparse
import time

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT


def used(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return v.total_bytes_allocated_per_bank * v.num_banks / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch4-steps", type=int, default=1000)
    args = ap.parse_args()
    with open_n300(trace_region_size=300_000_000) as mesh:
        m = Gemma4TT(mesh, Checkpoint(), max_batch=4)
        pts = m.allocate_kv_cache(4, 128 + args.steps + 1)
        ids = torch.randint(100, 20000, (128,))
        for b in range(4):
            m.prefill(ids, 0, pts[:, b])
        tok = torch.tensor([5])
        start = None
        t0 = time.time()
        for s in range(args.steps):
            lg = m.decode(tok, torch.tensor([128 + s]), pts[:, :1], trace=True)
            tok = lg.argmax(-1)
            if s == 10:
                start = used(mesh)
        b1 = used(mesh)
        print(f"b1: {args.steps} steps in {time.time() - t0:.0f} s, DRAM {start:.3f} -> {b1:.3f} GiB/chip", flush=True)
        tok4 = torch.full((4,), 5)
        for s in range(args.batch4_steps):
            lg = m.decode(tok4, torch.full((4,), 128 + s), pts, trace=True)  # rewrites positions the b1 run used
            tok4 = lg.argmax(-1)
            if s == 10:
                start4 = used(mesh)
        print(f"b4: {args.batch4_steps} steps, DRAM {start4:.3f} -> {used(mesh):.3f} GiB/chip", flush=True)
        m.release_traces()


if __name__ == "__main__":
    main()
