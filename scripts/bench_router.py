"""Decode router variants: topk_router_gpt vs linear + topk + softmax.

    pixi run python scripts/bench_router.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from tt_models.common.config import HIFI2, HIFI4, LOFI  # noqa: E402
from tt_models.common.mesh import open_n300  # noqa: E402

H, E, K = 2816, 128, 8


def main():
    with open_n300(trace_region_size=100_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        w = torch.randn(H, E) * 0.02
        xh = torch.randn(1, H)
        x = ttnn.from_torch(xh[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        wt = ttnn.from_torch(w[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        w2 = ttnn.reshape(wt, (H, E))
        bias = ttnn.from_torch(torch.zeros(1, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        ref = (xh.bfloat16().float() @ w.bfloat16().float())[0]
        rv, ri = ref.topk(K)
        print("ref ids", ri.tolist(), "weights", torch.softmax(rv, 0).tolist()[:3])

        def gpt():
            return ttnn.experimental.topk_router_gpt(ttnn.reshape(x, (1, H)), weight_tensor=w2, bias_tensor=bias, k=K, num_experts=E)

        print("topk_router_gpt us", timeit(mesh, lambda: gpt()[0]))
        for name, cc in (("HiFi4", HIFI4), ("HiFi2", HIFI2), ("LoFi", LOFI)):
            print(f"linear {name} us", timeit(mesh, lambda: ttnn.linear(x, wt, compute_kernel_config=cc, memory_config=ttnn.L1_MEMORY_CONFIG)))
        logits = ttnn.linear(x, wt, compute_kernel_config=HIFI4, memory_config=ttnn.L1_MEMORY_CONFIG)
        print("logit err", (ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float()[0, 0, 0] - ref).abs().max().item())
        try:
            print("topk us", timeit(mesh, lambda: ttnn.topk(logits, k=K, dim=-1)[0]))
            v, i = ttnn.topk(logits, k=K, dim=-1)
            print("topk ids", ttnn.to_torch(ttnn.get_device_tensors(i)[0]).flatten()[:K].tolist(), v.shape, v.layout, i.dtype)
            print("softmax us", timeit(mesh, lambda: ttnn.softmax(v, dim=-1)))
            s = ttnn.softmax(v, dim=-1)
            print("softmax", ttnn.to_torch(ttnn.get_device_tensors(s)[0]).float().flatten()[:K].tolist())

            def full():
                lg = ttnn.linear(x, wt, compute_kernel_config=HIFI4, memory_config=ttnn.L1_MEMORY_CONFIG)
                vv, ii = ttnn.topk(lg, k=K, dim=-1)
                lg.deallocate(True)
                ss = ttnn.softmax(vv, dim=-1)
                vv.deallocate(True)
                ii.deallocate(True)
                return ss

            print("linear+topk+softmax us", timeit(mesh, full))
        except Exception as e:  # noqa: BLE001
            print("failed", str(e).splitlines()[0][:200])


if __name__ == "__main__":
    main()
