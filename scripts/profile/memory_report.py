"""Device DRAM use per chip at each stage of the serving configuration.

pixi run python scripts/profile/memory_report.py --users 4 --context 32768
"""

import argparse

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT


def report(mesh: ttnn.MeshDevice, stage: str):
    """Mesh buffers are allocated in lock step, so one view covers each chip."""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    total = v.total_bytes_per_bank * v.num_banks
    used = v.total_bytes_allocated_per_bank * v.num_banks
    free_block = v.largest_contiguous_bytes_free_per_bank * v.num_banks
    print(
        f"{stage:28s} per chip: {used / 2**30:6.2f} GiB used of {total / 2**30:.2f} "
        f"GiB, largest free {free_block / 2**30:.2f} GiB",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--prefill", type=int, default=8192)
    args = ap.parse_args()
    with open_n300(trace_region_size=100_000_000) as mesh:
        report(mesh, "mesh open")
        model = Gemma4TT(
            mesh, Checkpoint(), max_batch=args.users, max_seq_len=args.context
        )
        report(mesh, "weights")
        tables = model.allocate_kv_cache(args.users, args.context)
        report(mesh, f"kv {model.kv.num_blocks} blocks")
        model.decode(
            torch.zeros(args.users, dtype=torch.long),
            torch.zeros(args.users, dtype=torch.long),
            tables,
            trace=True,
        )
        report(mesh, f"decode trace B={args.users}")
        model.prefill(torch.randint(100, 1000, (args.prefill,)), 0, tables[:, 0])
        report(mesh, f"after prefill {args.prefill}")


if __name__ == "__main__":
    main()
