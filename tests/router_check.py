"""Device check of the batch-1 router op (boundary.Router) against topk_router_gpt.

    pixi run python tests/router_check.py [--layers 0 5 17]
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.checkpoint import Checkpoint  # noqa: E402
from gemma4_tt.mesh import open_n300  # noqa: E402
from gemma4_tt.tt.boundary import Router  # noqa: E402
from gemma4_tt.tt.common import WeightLoader, rms_norm  # noqa: E402
from gemma4_tt.tt.moe import FeedForward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    with open_n300(trace_region_size=50_000_000) as mesh:
        loader = WeightLoader(mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8"))
        router = Router(mesh, cfg.hidden_size, cfg.num_experts, cfg.top_k_experts)
        for layer in args.layers:
            ff = FeedForward(loader, ckpt, cfg, layer)
            ff.decode_constants(1)
            same, wdiff = 0, 0.0
            err_mine, err_gpt = [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                r = ttnn.from_torch(torch.randn(1, 1, 1, cfg.hidden_size) * 3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                n = rms_norm(r, None, ff.eps)
                ids_ref, w_ref = ff._route(n, 1)
                ids, wts = router.outputs()
                router(n, ff.router, ids, wts)
                for c in range(2):
                    a = ttnn.to_torch(ttnn.get_device_tensors(ids)[c]).long().flatten()[:8].tolist()
                    b = ttnn.to_torch(ttnn.get_device_tensors(ids_ref)[c]).long().flatten()[:8].tolist()
                    wa = dict(zip(a, ttnn.to_torch(ttnn.get_device_tensors(wts)[c]).float().flatten()[:8].tolist()))
                    wb = dict(zip(b, ttnn.to_torch(ttnn.get_device_tensors(w_ref)[c]).float().flatten()[:8].tolist()))
                    same += set(a) == set(b)
                    if c == 0:  # FP32 reference on the device's BF16 inputs
                        nh = ttnn.to_torch(ttnn.get_device_tensors(n)[0]).float()[0, 0, 0]
                        wh = ttnn.to_torch(ttnn.get_device_tensors(ff.router)[0]).float()[0, 0]
                        v, i = (nh @ wh).topk(8)
                        ref = dict(zip(i.tolist(), torch.softmax(v, 0).tolist()))
                        err_mine.append(max(abs(wa[k] - ref.get(k, 0)) for k in a))
                        err_gpt.append(max(abs(wb[k] - ref.get(k, 0)) for k in b))
                    wdiff = max(wdiff, max(abs(wa[k] - wb[k]) for k in set(a) & set(b)))
                    if seed == 0 and c == 0:
                        print(f"layer {layer} ids {a}\n           ref {b}\n weights {[round(wa[k], 4) for k in a]}\n     ref {[round(wb[k], 4) for k in a if k in wb]}")
            print(f"layer {layer}: same id sets {same}/{2 * args.seeds}, max weight diff {wdiff:.4f}", flush=True)
            print(f"  max |w - fp32 ref|: router op {max(err_mine):.4f}, topk_router_gpt {max(err_gpt):.4f}", flush=True)
        print("topk_router_gpt us", timeit(mesh, lambda: ff._route(n, 1)[0]))

        def mine():
            i, w = router.outputs()
            router(n, ff.router, i, w)
            w.deallocate(True)
            return i

        print("router op us", timeit(mesh, mine))


if __name__ == "__main__":
    main()
