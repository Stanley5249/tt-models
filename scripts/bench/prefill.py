"""Prefill latency outside vLLM.

pixi run python scripts/bench/prefill.py --lens 128 2048 8192
"""

import argparse
import time

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 2048])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    with open_n300() as mesh:
        model = Gemma4TT(mesh, Checkpoint(), max_batch=1)
        pt = model.allocate_kv_cache(1, max(args.lens))[:, 0]
        for n in args.lens:
            ids = torch.randint(100, 20000, (n,))
            model.prefill(ids, 0, pt)  # compile
            times = []
            for _ in range(args.repeat):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                model.prefill(ids, 0, pt)
                times.append(time.perf_counter() - t0)
            print(f"prefill {n}: {min(times):.3f} s (min of {args.repeat})", flush=True)


if __name__ == "__main__":
    main()
