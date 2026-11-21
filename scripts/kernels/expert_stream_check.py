"""Check and time the decode expert stream ops for 1-4 token rows.

    pixi run python scripts/kernels/expert_stream_check.py [--rows 4] [--layer 0] [--gate-up-readers 4] [--down-readers 2]

Real layer weights. The stream and the prefill expert-major path (indexed
sparse_matmul, an independent implementation) get the same normed rows; the
per-chip partial sums (before the all-reduce) are compared, then the stream is
timed by trace replay (tt_models.common.timing.device_time_us).
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
from tt_models.models.gemma4.expert_stream import ExpertStream, pad_down
from tt_models.models.gemma4.moe import FeedForward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--gate-up-readers", type=int, default=4)
    ap.add_argument("--down-readers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rows", type=int, default=1)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    with open_n300(trace_region_size=100_000_000) as mesh:
        loader = WeightLoader(
            mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8")
        )
        ff = FeedForward(loader, ckpt, cfg, args.layer)
        rows = args.rows
        ff.decode_constants(rows)
        p = f"layers.{args.layer}."
        st = ExpertStream(
            mesh,
            cfg.hidden_size,
            ff.expert_local,
            ff.top_k,
            ff.n_experts,
            gate_up_readers=args.gate_up_readers,
            down_readers=args.down_readers,
            rows=rows,
        )
        w_down = loader.put(
            f"L{args.layer}.exp_down_p{st.down_width}",
            lambda: pad_down(
                ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous(),
                st.down_width,
            )[None],
            ttnn.bfloat4_b,
            shard_dim=-2,
        )
        scale = ttnn.from_torch(
            ckpt.get(p + "router.per_expert_scale").float().reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(
            "gate_up cores",
            len(st.gu_assign),
            "down cores",
            len(st.down_assign),
            flush=True,
        )
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            res = torch.randn(1, 1, rows, cfg.hidden_size) * 3
            r = ttnn.from_torch(
                res,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            normed = rms_norm(r, None, ff.eps)
            x = ttnn.mul(normed, ff.pre_ff2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ref = ff._experts(x, normed)

            def route():
                ids, weights = ff._route(normed, rows)
                if rows == 1:
                    return ids, weights
                flat = ttnn.typecast(
                    ttnn.reshape(ttnn.typecast(ids, ttnn.uint32), (1, rows * ff.top_k)),
                    ttnn.uint16,
                )
                return flat, ttnn.reshape(weights, (1, rows * ff.top_k))

            ids, weights = route()
            if seed == 0:
                print(
                    "ids",
                    ids.memory_config().buffer_type,
                    ids.shape,
                    "weights",
                    weights.memory_config().buffer_type,
                    flush=True,
                )

            def stream():
                act = st.act_tensor()
                st.gate_up(x, ids, weights, scale, ff.exp_gate_up, act)
                out = st.out_tensor(ttnn.L1_MEMORY_CONFIG)
                st.down(act, ids, w_down, out)
                act.deallocate(True)
                return out

            out = stream()
            for c, (a, b) in enumerate(
                zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(ref))
            ):
                for t in range(rows):
                    ta, tb = (
                        ttnn.to_torch(a).float()[0, 0, t],
                        ttnn.to_torch(b).float()[0, 0, t, : cfg.hidden_size],
                    )
                    pcc = torch.corrcoef(torch.stack([ta, tb]))[0, 1].item()
                    print(
                        f"seed {seed} chip {c} row {t}: pcc {pcc:.6f}, max diff {(ta - tb).abs().max():.4f}, ref max {tb.abs().max():.3f}",
                        flush=True,
                    )
            out.deallocate(True)

        def routed_stream():
            i, w = route()
            o = stream()
            i.deallocate(True)
            w.deallocate(True)
            return o

        print(
            "stream path (router + 2 ops) us",
            device_time_us(mesh, routed_stream, iters=20),
        )
        print("router alone us", device_time_us(mesh, lambda: route()[0], iters=20))

        def gu_only():
            act = st.act_tensor()
            return st.gate_up(x, ids, weights, scale, ff.exp_gate_up, act)

        print("gate_up alone us", device_time_us(mesh, gu_only, iters=20))
        act = gu_only()
        print(
            "down alone us",
            device_time_us(
                mesh,
                lambda: st.down(act, ids, w_down, st.out_tensor(ttnn.L1_MEMORY_CONFIG)),
                iters=20,
            ),
        )


if __name__ == "__main__":
    main()
