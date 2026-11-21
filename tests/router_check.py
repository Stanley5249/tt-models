"""Device check of the router op (boundary.Router) against topk_router_gpt, for 1-4 token rows.

    pixi run python tests/router_check.py [--layers 0 5 17] [--rows 1 2 3 4]
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


def host(t, c=0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    k = cfg.top_k_experts
    with open_n300(trace_region_size=50_000_000) as mesh:
        loader = WeightLoader(mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8"))
        for rows in args.rows:
            router = Router(mesh, cfg.hidden_size, cfg.num_experts, k, rows=rows)
            for layer in args.layers:
                ff = FeedForward(loader, ckpt, cfg, layer)
                ff.decode_constants(rows)
                same, total, wdiff = 0, 0, 0.0
                err_mine, err_gpt = [], []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    r = ttnn.from_torch(torch.randn(1, 1, rows, cfg.hidden_size) * 3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    n = rms_norm(r, None, ff.eps)
                    ids_ref, w_ref = ff._route(n, rows)  # [rows, k]
                    ids, wts = router.outputs()
                    router(n, ff.router, ids, wts)
                    nh = host(n).float()[0, 0, :rows]
                    wh = host(ff.router).float()[0, 0]
                    for c in range(2):
                        got_i = host(ids, c).long().flatten()[: rows * k].reshape(rows, k)
                        got_w = host(wts, c).float().flatten()[: rows * k].reshape(rows, k)
                        ref_i = host(ids_ref, c).long().reshape(-1, k)[:rows]
                        ref_w = host(w_ref, c).float().reshape(-1, k)[:rows]
                        for b in range(rows):
                            a, bb = got_i[b].tolist(), ref_i[b].tolist()
                            wa, wb = dict(zip(a, got_w[b].tolist())), dict(zip(bb, ref_w[b].tolist()))
                            same += set(a) == set(bb)
                            total += 1
                            wdiff = max(wdiff, max(abs(wa[e] - wb[e]) for e in set(a) & set(bb)))
                            if c == 0:  # FP32 reference on the device's BF16 inputs
                                v, i = (nh[b] @ wh).topk(k)
                                ref = dict(zip(i.tolist(), torch.softmax(v, 0).tolist()))
                                err_mine.append(max(abs(wa[e] - ref.get(e, 0)) for e in a))
                                err_gpt.append(max(abs(wb[e] - ref.get(e, 0)) for e in bb))
                print(f"rows {rows} layer {layer}: same id sets {same}/{total}, max weight diff {wdiff:.4f}; "
                      f"max |w - fp32 ref| router op {max(err_mine):.4f}, topk_router_gpt {max(err_gpt):.4f}", flush=True)

            def mine():
                i, w = router.outputs()
                router(n, ff.router, i, w)
                w.deallocate(True)
                return i

            print(f"rows {rows}: topk_router_gpt us {timeit(mesh, lambda: ff._route(n, rows)[0])}, router op us {timeit(mesh, mine)}", flush=True)


if __name__ == "__main__":
    main()
