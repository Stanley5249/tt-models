"""Spread the decode expert sparse matmuls over more cores.

    pixi run python scripts/bench_sparse_ksplit.py

sparse_matmul splits only N over cores and loops over the selected experts on
every core, so the per-chip gate_up (N = 24 tiles) runs on 24 cores and the
down (N = 88 tiles) on 30. Each core reads tile by tile at a few GB/s.

gate_up K-split by s: expert e's [K, N] weight is stored as [K/s, s*N], the
K-chunks side by side, and x enters as s rows [x_0; ...; x_{s-1}] of K/s. Row q
times column block q is x_q @ W_q, so the result is the sum of the diagonal
blocks. Same weight bytes, s times the N tiles. Down: N padded 2816 -> 3072
so 48 or 32 cores form a full rectangle (the 1D mcast config needs one).
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.mesh import open_n300  # noqa: E402

E, H, N = 128, 2816, 768
LOFI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)


def cfg(per_core_n, grid, in0_block_w, m_tiles=1):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=m_tiles, out_block_w=per_core_n, per_core_M=m_tiles, per_core_N=per_core_n, fuse_batch=False,
        fused_activation=None, mcast_in0=True)


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dev(t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep)

        ids = [3, 17, 40, 51, 77, 90, 101, 126]
        idx = dev(torch.tensor([ids], dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        sp = dev(torch.ones(1, 1, 1, E), lay=ttnn.ROW_MAJOR_LAYOUT)
        torch.manual_seed(0)
        w_full = torch.randn(E, H, N) * 0.02
        x_full = torch.randn(H)
        ref = torch.stack([x_full @ w_full[e] for e in ids])  # [8, N]
        for s in (1, 2, 4):
            k = H // s
            w = w_full.reshape(E, s, k, N).permute(0, 2, 1, 3).reshape(E, k, s * N)  # [E, K/s, s*N]
            b = dev(w[None], ttnn.bfloat4_b)
            x = dev(x_full.reshape(s, k)[None, None])
            nt = s * N // 32
            for pc, grid in ((1, (8, 3)), (1, (8, 6)), (2, (8, 6)), (3, (8, 4)), (2, (8, 3))):
                if -(-nt // pc) != grid[0] * grid[1]:
                    continue
                for blk in (22, 44, 11):
                    if (k // 32) % blk:
                        continue
                    c = cfg(pc, grid, blk)
                    kw = dict(sparsity=sp, indices=idx, program_config=c, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                    try:
                        out = ttnn.to_torch(ttnn.get_device_tensors(ttnn.sparse_matmul(x, b, **kw))[0]).float().reshape(8, -1, s * N)
                        got = sum(out[:, q, q * N : (q + 1) * N] for q in range(s))
                        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                        us = timeit(mesh, lambda: ttnn.sparse_matmul(x, b, **kw))
                        mb = 8 * H * N * 0.5625 / 1e6
                        print(f"gate_up s={s} cores={grid[0] * grid[1]} per_core_n={pc} in0_block_w={blk}: {us} us, {mb / us * 1e3:.0f} GB/s, pcc {pcc:.5f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"gate_up s={s} pc={pc} grid={grid} blk={blk}: failed {str(e).splitlines()[0][:120]}", flush=True)
            b.deallocate(True)
        for width in (2816, 3072):
            b = dev(torch.randn(1, E, 352, width) * 0.02, ttnn.bfloat4_b)
            a = dev(torch.randn(1, 8, 1, 352))
            nt = width // 32
            for pc, grid in ((3, (6, 5)), (2, (8, 6)), (3, (8, 4)), (4, (8, 3))):
                if -(-nt // pc) != grid[0] * grid[1]:
                    continue
                c = cfg(pc, grid, 11)
                kw = dict(sparsity=sp, indices=idx, program_config=c, compute_kernel_config=LOFI, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          dtype=ttnn.bfloat16, is_input_a_sparse=True, is_input_b_sparse=True)
                try:
                    us = timeit(mesh, lambda: ttnn.sparse_matmul(a, b, **kw))
                    mb = 8 * 352 * H * 0.5625 / 1e6
                    print(f"down N={width} cores={grid[0] * grid[1]} per_core_n={pc}: {us} us, {mb / us * 1e3:.0f} GB/s (real bytes)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"down N={width} pc={pc}: failed {str(e).splitlines()[0][:120]}", flush=True)
            b.deallocate(True)


if __name__ == "__main__":
    main()
