"""Device check of the router op (router.Router) against topk_router_gpt, for 1-16
token rows.

pixi run python scripts/kernels/router_check.py [--layers 0 5 17] [--rows 1 2 4 16]
"""

import argparse
from pathlib import Path

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.common.norm import rms_norm
from tt_models.common.timing import device_time_us
from tt_models.common.weights import WeightLoader
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.moe import FeedForward
from tt_models.models.gemma4.router import Router


def host(t: ttnn.Tensor, c: int = 0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    assert cfg.num_experts is not None
    assert cfg.top_k_experts is not None
    k = cfg.top_k_experts
    with open_n300(trace_region_size=50_000_000) as mesh:
        loader = WeightLoader(
            mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8")
        )
        for rows in args.rows:
            router = Router(mesh, cfg.hidden_size, cfg.num_experts, k, rows=rows)
            for layer in args.layers:
                ff = FeedForward(loader, ckpt, cfg, layer)
                ff.decode_constants(rows)
                same, total, wdiff = 0, 0, 0.0
                err_mine: list[float] = []
                err_gpt: list[float] = []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    r = ttnn.from_torch(
                        torch.randn(1, 1, rows, cfg.hidden_size) * 3,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    )
                    n = rms_norm(r, None, ff.eps)
                    ids_ref, w_ref = ff._route(n, rows)  # [rows, k]
                    ids, wts = router.outputs()
                    router(n, ff.router, ids, wts)
                    nh = host(n).float()[0, 0, :rows]
                    wh = host(ff.router).float()[0, 0]
                    for c in range(2):
                        got = router.decode(host(ids, c), host(wts, c))
                        ref_i = host(ids_ref, c).long().reshape(-1, k)[:rows]
                        ref_w = host(w_ref, c).float().reshape(-1, k)[:rows]
                        for b in range(rows):
                            wa, bb = got[b], ref_i[b].tolist()
                            a = list(wa)
                            wb = dict(zip(bb, ref_w[b].tolist()))
                            same += set(a) == set(bb)
                            total += 1
                            wdiff = max(
                                wdiff, *(abs(wa[e] - wb[e]) for e in set(a) & set(bb))
                            )
                            if c == 0:  # FP32 reference on the device's BF16 inputs
                                v, i = (nh[b] @ wh).topk(k)
                                ref = dict(
                                    zip(i.tolist(), torch.softmax(v, 0).tolist())
                                )
                                err_mine.append(
                                    max(abs(wa[e] - ref.get(e, 0)) for e in a)
                                )
                                err_gpt.append(
                                    max(abs(wb[e] - ref.get(e, 0)) for e in bb)
                                )
                print(
                    f"rows {rows} layer {layer}: same id sets {same}/{total}, max "
                    f"weight diff {wdiff:.4f}; "
                    f"max |w - fp32 ref| router op {max(err_mine):.4f}, "
                    f"topk_router_gpt {max(err_gpt):.4f}",
                    flush=True,
                )

            def mine():
                i, w = router.outputs()
                router(n, ff.router, i, w)
                w.deallocate(True)
                return i

            print(
                f"rows {rows}: topk_router_gpt us "
                f"{device_time_us(mesh, lambda: ff._route(n, rows)[0])}, "
                f"router op us {device_time_us(mesh, mine)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
