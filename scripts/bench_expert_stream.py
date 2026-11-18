"""Check and time the batch-1 expert stream ops against the sparse_matmul path.

    pixi run python scripts/bench_expert_stream.py [--layer 0] [--gate-up-readers 4] [--down-readers 2]

Real layer weights. Both paths get the same normed input and routing; the
per-chip partial sums (before the all-reduce) are compared, then each path is
timed by trace replay (scripts/bench_norm.timeit).
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from gemma4_tt.checkpoint import Checkpoint  # noqa: E402
from gemma4_tt.mesh import open_n300  # noqa: E402
from gemma4_tt.tt.common import WeightLoader, rms_norm  # noqa: E402
from gemma4_tt.tt.expert_stream import ExpertStream, pad_down  # noqa: E402
from gemma4_tt.tt.moe import FeedForward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--gate-up-readers", type=int, default=4)
    ap.add_argument("--down-readers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    with open_n300(trace_region_size=100_000_000) as mesh:
        loader = WeightLoader(mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8"))
        ff = FeedForward(loader, ckpt, cfg, args.layer)
        ff.decode_constants(1)
        p = f"layers.{args.layer}."
        st = ExpertStream(mesh, cfg.hidden_size, ff.expert_local, ff.top_k, ff.n_experts, args.gate_up_readers, args.down_readers)
        w_down = loader.put(
            f"L{args.layer}.exp_down_p{st.down_width}",
            lambda: pad_down(ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous(), st.down_width)[None],
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
        print("gate_up cores", len(st.gu_assign), "down cores", len(st.down_assign), flush=True)
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            res = torch.randn(1, 1, 1, cfg.hidden_size) * 3
            r = ttnn.from_torch(res, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            normed = rms_norm(r, None, ff.eps)
            x = ttnn.mul(normed, ff.pre_ff2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ref = ff._experts_decode(x, normed)

            def route():
                return ttnn.experimental.topk_router_gpt(
                    ttnn.reshape(normed, (1, cfg.hidden_size)), weight_tensor=ff.router_2d, bias_tensor=ff._bias(1), k=ff.top_k,
                    num_experts=ff.n_experts,
                )

            ids, weights = route()
            if seed == 0:
                print("ids", ids.memory_config().buffer_type, ids.shape, "weights", weights.memory_config().buffer_type, flush=True)

            def stream():
                act = st.act_tensor()
                st.gate_up(x, ids, weights, scale, ff.exp_gate_up, act)
                out = st.out_tensor(ttnn.L1_MEMORY_CONFIG)
                st.down(act, ids, w_down, out)
                act.deallocate(True)
                return out

            out = stream()
            for c, (a, b) in enumerate(zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(ref))):
                ta, tb = ttnn.to_torch(a).float()[0, 0, 0], ttnn.to_torch(b).float()[0, 0, 0]
                pcc = torch.corrcoef(torch.stack([ta, tb]))[0, 1].item()
                print(f"seed {seed} chip {c}: pcc {pcc:.6f}, max diff {(ta - tb).abs().max():.4f}, ref max {tb.abs().max():.3f}", flush=True)
            out.deallocate(True)
        print("sparse path us", timeit(mesh, lambda: ff._experts_decode(x, normed), iters=20))

        def routed_stream():
            i, w = route()
            o = stream()
            i.deallocate(True)
            w.deallocate(True)
            return o

        print("stream path (router + 2 ops) us", timeit(mesh, routed_stream, iters=20))
        print("router alone us", timeit(mesh, lambda: route()[0], iters=20))

        def gu_only():
            act = st.act_tensor()
            return st.gate_up(x, ids, weights, scale, ff.exp_gate_up, act)

        print("gate_up alone us", timeit(mesh, gu_only, iters=20))
        act = gu_only()
        print("down alone us", timeit(mesh, lambda: st.down(act, ids, w_down, st.out_tensor(ttnn.L1_MEMORY_CONFIG)), iters=20))


if __name__ == "__main__":
    main()
