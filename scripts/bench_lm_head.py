"""LM head matmul variants on the real per-chip [2816, 65536] BFP8 weight.

    pixi run python scripts/bench_lm_head.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from tt_models.common.config import HIFI2, LOFI, compute_config  # noqa: E402
from tt_models.common.mesh import open_n300  # noqa: E402


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        w = ttnn.as_tensor(None, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name="cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8/lm_head")
        x = ttnn.from_torch(torch.randn(1, 1, 32, 2816), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        xl1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        dram = ttnn.DRAM_MEMORY_CONFIG
        hifi2_noacc = compute_config(ttnn.MathFidelity.HiFi2, packer_l1_acc=False)
        cases = {
            "model (fp32 out, HiFi2)": lambda: ttnn.linear(x, w, compute_kernel_config=HIFI2, memory_config=dram, dtype=ttnn.float32),
            "bf16 out, HiFi2": lambda: ttnn.linear(x, w, compute_kernel_config=HIFI2, memory_config=dram),
            "fp32 out, LoFi": lambda: ttnn.linear(x, w, compute_kernel_config=LOFI, memory_config=dram, dtype=ttnn.float32),
            "fp32 out, HiFi2, L1 in": lambda: ttnn.linear(xl1, w, compute_kernel_config=HIFI2, memory_config=dram, dtype=ttnn.float32),
            "fp32 out, HiFi2, no l1 acc": lambda: ttnn.linear(x, w, compute_kernel_config=hifi2_noacc, memory_config=dram, dtype=ttnn.float32),
        }
        for k_blk in (4, 8, 11, 22):
            for per_n in (32, 64):
                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), in0_block_w=k_blk, out_subblock_h=1, out_subblock_w=4 if per_n % 4 == 0 else 1,
                    out_block_h=1, out_block_w=per_n, per_core_M=1, per_core_N=per_n, fuse_batch=True, fused_activation=None, mcast_in0=True)
                cases[f"1D k_blk={k_blk} per_core_N={per_n}"] = (lambda c: lambda: ttnn.linear(
                    x, w, program_config=c, compute_kernel_config=HIFI2, memory_config=dram, dtype=ttnn.float32))(cfg)
        for name, fn in cases.items():
            try:
                print(f"{name}: {timeit(mesh, fn, iters=10)} us", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: failed {str(e).splitlines()[0][:150]}", flush=True)


if __name__ == "__main__":
    main()
