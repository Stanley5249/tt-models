"""Microbenchmark the decode expert sparse matmuls (indexed, 8 experts) over program configs.

    pixi run python scripts/bench_sparse_matmul.py
"""

import time

import torch
import ttnn

from gemma4_tt.mesh import open_n300

E, H = 128, 2816


def config(n_tiles, per_core_n, grid, in0_block_w):
    cores = -(-n_tiles // per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    ), cores


def run(mesh, a, b, idx, sp, cfg, a_sparse, iters=20):
    kw = dict(sparsity=sp, indices=idx, program_config=cfg, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
              compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True))
    if a_sparse:
        kw.update(is_input_a_sparse=True, is_input_b_sparse=True)
    ttnn.sparse_matmul(a, b, **kw).deallocate(True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.sparse_matmul(a, b, **kw).deallocate(True)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    with open_n300() as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dev(t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep)

        idx = dev(torch.tensor([[3, 17, 40, 51, 77, 90, 101, 126]], dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        sp = dev(torch.ones(1, 1, 1, E), lay=ttnn.ROW_MAJOR_LAYOUT)
        x = dev(torch.randn(1, 1, 1, H))
        for width in (704, 768):  # per-chip gate|up width, unpadded and padded
            b = dev(torch.randn(1, E, H, width) * 0.02, ttnn.bfloat4_b)
            nt = width // 32
            for pc, grid in ((3, (8, 1)), (1, (8, 3)), (2, (6, 2)), (1, (2, 11))):
                if -(-nt // pc) != grid[0] * grid[1] or grid[1] > 8:
                    continue
                for w in (2, 4, 8, 22, 44):
                    if 88 % w:
                        continue
                    cfg, cores = config(nt, pc, grid, w)
                    try:
                        us = run(mesh, x, b, idx, sp, cfg, False)
                        mb = 8 * H * width * 0.5625 / 1e6
                        print(f"gate_up N={width} cores={cores} per_core_n={pc} in0_block_w={w}: {us:.1f} us, {mb / us * 1e3:.0f} GB/s", flush=True)
                    except Exception as e:
                        print(f"gate_up N={width} pc={pc} w={w}: {str(e)[:100]}")
            b.deallocate(True)
        for k in (352, 384):
            b = dev(torch.randn(1, E, k, H) * 0.02, ttnn.bfloat4_b)
            a = dev(torch.randn(1, 8, 1, k))
            kt = k // 32
            for pc, grid in ((3, (6, 5)), (2, (4, 11)), (4, (2, 11)), (2, (8, 6)), (11, (8, 1))):
                if -(-88 // pc) != grid[0] * grid[1] or grid[0] > 8 or grid[1] > 8:
                    continue
                for w in (1, kt // 2 if kt % 2 == 0 else 1, kt, 4, 3):
                    if kt % w:
                        continue
                    cfg, cores = config(88, pc, grid, w)
                    try:
                        us = run(mesh, a, b, idx, sp, cfg, True)
                        mb = 8 * k * H * 0.5625 / 1e6
                        print(f"down K={k} cores={cores} per_core_n={pc} in0_block_w={w}: {us:.1f} us, {mb / us * 1e3:.0f} GB/s", flush=True)
                    except Exception as e:
                        print(f"down K={k} pc={pc} w={w}: {str(e)[:100]}")
            b.deallocate(True)


if __name__ == "__main__":
    main()
