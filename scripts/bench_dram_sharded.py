"""Microbenchmark decode dense matmuls: current 1D/default configs versus DRAM-sharded weights.

    pixi run python scripts/bench_dram_sharded.py

Device time per call from a replayed trace (scripts/bench_norm.py:timeit). The
DRAM-sharded variant takes a width-sharded L1 activation and returns a
width-sharded output; "+s2i" adds the conversion back to interleaved that the
following ops need.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.mesh import open_n300  # noqa: E402

TILE = 32
HIFI2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
SHAPES = {  # name: (K, N) per chip
    "wqkv sliding": (2816, 4096),
    "wqkv global": (2816, 5120),
    "wo sliding": (2048, 2816),
    "wo global": (4096, 2816),
    "mlp gate_up": (2816, 2112),
    "mlp gate_up pad": (2816, 2304),
    "mlp down": (1056, 2816),
    "mlp down pad": (1152, 2816),
    "lm_head quarter": (2816, 16384),
}


def width_sharded(rows, width, cores):
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // cores),
        core_grid=ttnn.CoreGrid(x=cores, y=1) if cores <= 8 else ttnn.CoreGrid(x=8, y=cores // 8),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )


def dram_weight_config(mesh, k, n):
    banks = mesh.dram_grid_size().x
    padded = -(-n // (TILE * banks)) * TILE * banks
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        for name, (k, n) in SHAPES.items():
            w = torch.randn(1, 1, k, n) * 0.02
            x = ttnn.from_torch(torch.randn(1, 1, 32, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            w_il = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            base = timeit(mesh, lambda: ttnn.linear(x, w_il, compute_kernel_config=HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            w_il.deallocate(True)
            w_ds = ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=dram_weight_config(mesh, k, n)
            )
            line = f"{name:16s} K={k} N={n}: interleaved {base} us"
            for cores in (8, 4, 2):
                if (k // TILE) % cores or (n // TILE) % cores:
                    continue
                x_sh = ttnn.to_memory_config(x, width_sharded(32, k, cores))
                cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=k // TILE // cores, per_core_M=1, per_core_N=n // TILE // cores, fused_activation=None
                )
                out_cfg = width_sharded(32, n, cores)

                def ds():
                    return ttnn.linear(x_sh, w_ds, program_config=cfg, compute_kernel_config=HIFI2, memory_config=out_cfg)

                def ds_s2i():
                    y = ds()
                    out = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
                    y.deallocate(True)
                    return out

                try:
                    line += f" | dram-sharded {cores} cores {timeit(mesh, ds)} us, +s2i {timeit(mesh, ds_s2i)} us"
                except Exception as e:  # noqa: BLE001
                    line += f" | {cores} cores failed: {str(e)[:120]}"
                x_sh.deallocate(True)
            print(line, flush=True)
            w_ds.deallocate(True)


if __name__ == "__main__":
    main()
