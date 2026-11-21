"""All-reduce latency against payload size, and the cost of packing one row.

    pixi run python scripts/bench_ccl_small.py

A decode all-reduce sums one real row padded to a [32, W] tile. A sum is
elementwise, so the row can be packed into a dense [32, W/32] block first.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from tt_models.common.mesh import open_n300  # noqa: E402


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        dram = ttnn.DRAM_MEMORY_CONFIG
        for rows, width, layout in ((32, 32, "tile"), (32, 96, "tile"), (32, 192, "tile"), (32, 704, "tile"), (32, 2816, "tile"),
                                    (1, 2816, "rm"), (1, 5632, "rm")):
            lay = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
            host = torch.randn(2, 1, rows, width)
            x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=lay, device=mesh, mesh_mapper=shard)
            try:
                t = timeit(mesh, lambda: ttnn.all_reduce(x, cluster_axis=1, memory_config=dram))
                out = ttnn.to_torch(ttnn.get_device_tensors(ttnn.all_reduce(x, cluster_axis=1, memory_config=dram))[0]).float()
                err = (out[0] - host.sum(0)).abs().max().item()
                print(f"all_reduce [{rows}, {width}] {layout}: {t} us, max err {err:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"all_reduce [{rows}, {width}] {layout}: failed {str(e).splitlines()[0][:150]}", flush=True)
            x.deallocate(True)
        for width in (2816, 5632):
            host = torch.randn(2, 1, 32, width)
            host[:, :, 1:] = 0
            x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)

            def packed():
                p = ttnn.reshape(ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, width]), (1, 1, 32, width // 32))
                r = ttnn.all_reduce(p, cluster_axis=1, memory_config=dram)
                p.deallocate(True)
                out = ttnn.reshape(r, (1, 1, 1, width))
                return out

            print(f"pack only {width}:", timeit(mesh, lambda: ttnn.reshape(ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, width]), (1, 1, 32, width // 32))), flush=True)
            try:
                out = ttnn.to_torch(ttnn.get_device_tensors(packed())[0]).float()
                err = (out[0, 0, 0] - host.sum(0)[0, 0]).abs().max().item()
                print(f"packed all_reduce {width}: {timeit(mesh, packed)} us, max err {err:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"packed {width}: failed {str(e).splitlines()[0][:150]}", flush=True)


if __name__ == "__main__":
    main()
