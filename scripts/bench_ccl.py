"""Microbenchmark decode all-reduce variants across the two N300 chips.

    pixi run python scripts/bench_ccl.py

Inputs are [1, 1, 32, W] BF16 tiles (one decode row padded to a tile), as the
attention (W = 2816) and feed-forward (W = 5632) all-reduces see them. Times
are device time per call from trace replay (bench_norm.timeit).

ttnn.experimental.all_reduce_async (ttnn 0.81, FABRIC_1D) hung the Ethernet
cores here and needed a card reset, so it is not in the list; so did ttnn.all_reduce with num_links=2.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.mesh import open_n300  # noqa: E402


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        for width in (2816, 5632):
            host = torch.randn(2, 1, 32, width)
            x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
            ref = host.sum(0, keepdim=True)
            results = {}

            def check(name, fn):
                try:
                    out = fn()
                    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[..., :32, :width]
                    out.deallocate(True)
                    err = (got[0, 0, :1] - ref[0, 0, :1]).abs().max().item()
                    results[name] = (timeit(mesh, fn), round(err, 3))
                except Exception as e:  # noqa: BLE001
                    results[name] = f"failed: {str(e).splitlines()[0][:120]}"
                print(width, name, results[name], flush=True)

            check("all_reduce", lambda: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            check("all_reduce L1", lambda: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG))

            def gather_sum(mem):
                g = ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=mem)
                out = ttnn.experimental.fast_reduce_nc(g, dims=[0], memory_config=mem)
                g.deallocate(True)
                return out

            check("all_gather+reduce DRAM", lambda: gather_sum(ttnn.DRAM_MEMORY_CONFIG))
            check("all_gather+reduce L1", lambda: gather_sum(ttnn.L1_MEMORY_CONFIG))

            def gather_add():
                g = ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG)
                a = ttnn.slice(g, [0, 0, 0, 0], [1, 1, 32, width])
                b = ttnn.slice(g, [1, 0, 0, 0], [2, 1, 32, width])
                out = ttnn.add(a, b)
                for t in (g, a, b):
                    t.deallocate(True)
                return out

            check("all_gather+slice+add", gather_add)
            check("all_gather only", lambda: ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG))
            check("reduce_scatter only", lambda: ttnn.reduce_scatter(x, dim=3, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG))
            x.deallocate(True)


if __name__ == "__main__":
    main()
