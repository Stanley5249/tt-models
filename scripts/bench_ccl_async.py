"""Decode all-reduce with the semaphore-based async CCL ops, as tt_transformers does.

    pixi run python scripts/bench_ccl_async.py

Compares against ttnn.all_reduce on [1, 1, 32, W] tiles. Uses one link and
Linear topology only (num_links=2 hung the card with ttnn.all_reduce).
"""

import sys
from pathlib import Path

import torch
import ttnn
from models.tt_transformers.tt.ccl import TT_CCL

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.mesh import open_n300  # noqa: E402

LIN = ttnn.Topology.Linear


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        ccl = TT_CCL(mesh)
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        for width in (2816, 5632):
            for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
                host = torch.randn(2, 1, 32, width)
                x = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
                ref = host.sum(0)[0, 0]

                def rs_ag(mem, workers=2):
                    r = ttnn.experimental.reduce_scatter_minimal_async(
                        x, persistent_output_buffers=None, dim=3,
                        multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
                        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                        num_links=1, cluster_axis=1, memory_config=mem, intermediate_memory_config=mem, topology=LIN,
                        chunks_per_sync=10, num_workers_per_link=workers, num_buffers_per_channel=2,
                    )
                    out = ttnn.experimental.all_gather_async(
                        r, persistent_output_buffer=None, dim=3,
                        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                        num_links=1, cluster_axis=1, topology=LIN, memory_config=mem,
                        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                        chunks_per_sync=10, num_workers_per_link=workers, num_buffers_per_channel=2,
                    )
                    r.deallocate(True)
                    return out

                def ag_sum(mem):
                    g = ttnn.experimental.all_gather_async(
                        x, persistent_output_buffer=None, dim=0,
                        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                        num_links=1, cluster_axis=1, topology=LIN, memory_config=mem,
                        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                        chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2,
                    )
                    out = ttnn.experimental.fast_reduce_nc(g, dims=[0], memory_config=mem)
                    g.deallocate(True)
                    return out

                variants = {
                    "ttnn.all_reduce": lambda: ttnn.all_reduce(x, cluster_axis=1, memory_config=dram),
                    "rs+ag async DRAM": lambda: rs_ag(dram),
                    "rs+ag async L1": lambda: rs_ag(l1),
                    "rs+ag async L1 1 worker": lambda: rs_ag(l1, 1),
                    "ag async + sum L1": lambda: ag_sum(l1),
                }
                for name, fn in variants.items():
                    try:
                        out = fn()
                        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, 0]
                        out.deallocate(True)
                        err = (got - ref).abs().max().item()
                        print(f"{width} {dtype} {name}: {timeit(mesh, fn)} us, max err {err:.3f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{width} {dtype} {name}: failed {str(e).splitlines()[0][:150]}", flush=True)
                x.deallocate(True)


if __name__ == "__main__":
    main()
