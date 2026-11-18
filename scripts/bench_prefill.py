"""Prefill latency outside vLLM.

    pixi run python scripts/bench_prefill.py --lens 128 2048 8192
"""

import argparse
import time

import torch
import ttnn

from gemma4_tt.checkpoint import Checkpoint
from gemma4_tt.mesh import open_n300
from gemma4_tt.tt.model import Gemma4TT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 2048])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    with open_n300() as mesh:
        model = Gemma4TT(mesh, Checkpoint(), max_batch=1)
        blocks = -(-max(args.lens) // model.block_size) + 1
        model.allocate_kv_cache(blocks + 1)
        pt = torch.arange(blocks, dtype=torch.int32)
        for n in args.lens:
            ids = torch.randint(100, 20000, (n,))
            model.prefill(ids, pt, slot=0)  # compile
            times = []
            for _ in range(args.repeat):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                model.prefill(ids, pt, slot=0)
                times.append(time.perf_counter() - t0)
            print(f"prefill {n}: {min(times):.3f} s (min of {args.repeat})", flush=True)


if __name__ == "__main__":
    main()
