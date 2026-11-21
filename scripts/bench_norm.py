"""Microbenchmark decode RMSNorm variants on [1, 1, 32, 2816].

    pixi run python scripts/bench_norm.py
"""

import time

import torch
import ttnn

from tt_models.common.mesh import open_n300

H = 2816


def timeit(mesh, fn, iters=50):
    """Device time per call: replay a trace of `iters` calls, so host dispatch is excluded."""
    fn().deallocate(True)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        fn().deallocate(True)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / iters * 1e6
    ttnn.release_trace(mesh, tid)
    return round(dt, 1)


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        x = ttnn.from_torch(torch.randn(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        w_rm = ttnn.from_torch(torch.randn(1, 1, H // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        w_tile = ttnn.from_torch(torch.randn(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        hifi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        print("interleaved DRAM", timeit(mesh, lambda: ttnn.rms_norm(x, weight=w_rm, epsilon=1e-6, compute_kernel_config=hifi)))
        print("interleaved DRAM HiFi2", timeit(mesh, lambda: ttnn.rms_norm(x, weight=w_rm, epsilon=1e-6)))
        print("mul by weight", timeit(mesh, lambda: ttnn.mul(x, w_tile)))
        for cores, grid in ((22, (2, 11)), (11, (1, 11)), (44, (4, 11)), (8, (8, 1)), (16, (8, 2)), (32, (8, 4))):
            if grid[1] > 8:
                grid = (grid[1], grid[0]) if grid[1] <= 8 else None
            if grid is None or grid[0] > 8:
                continue
            per = H // 32 // cores
            if per * cores * 32 != H:
                continue
            shard = ttnn.create_sharded_memory_config(
                shape=(32, per * 32), core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]), strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid, subblock_w=next(w for w in (4, 2, 1) if per % w == 0), block_h=1, block_w=per, inplace=False
            )

            def fn():
                xs = ttnn.to_memory_config(x, shard)
                y = ttnn.rms_norm(xs, weight=w_rm, epsilon=1e-6, program_config=cfg, memory_config=shard, compute_kernel_config=hifi)
                xs.deallocate(True)
                out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
                y.deallocate(True)
                return out

            try:
                print(f"width-sharded {cores} cores incl. conversions", timeit(mesh, fn))
            except Exception as e:
                print(cores, str(e)[:120])


if __name__ == "__main__":
    main()
