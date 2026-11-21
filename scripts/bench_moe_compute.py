"""Feasibility check of ttnn.experimental.moe_compute for Gemma 4 experts.

    pixi run python scripts/bench_moe_compute.py [--tokens 1 4]

moe_compute needs a replicated expert mapping (every expert on one chip), so
this is expert parallel: each chip holds 64 full-width experts (N = 704) and
writes the [k, T, H] rows of the (token, slot) pairs routed to its experts; the
caller would sum the rows and all-reduce across chips. Both chips hold the same
64 random experts here (expert e uses weights e % 64), which is enough for
timing and for checking each chip's own rows.

Reports device time per call (trace replay) and the PCC of each chip's rows
against a torch reference that uses tanh GELU, as Gemma does.
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn
from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)
from ttnn.operations.ccl import MoEActivationFunction

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from tt_models.common.mesh import open_n300  # noqa: E402

H, N, E, K = 2816, 704, 128, 8
LOCAL = E // 2
OUT_H_SHARDS = 4


def l1_on_core(core, shape):
    spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}), list(shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()
    e = LOCAL
    torch.manual_seed(0)
    with open_n300(trace_region_size=200_000_000) as mesh:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        w0 = torch.randn(1, e, H, N) * 0.02
        w1 = torch.randn(1, e, H, N) * 0.02
        w2 = torch.randn(1, e, N, H) * 0.02
        s01, s2, dram = get_weight_core_shard_maps(mesh, H, N)
        mc01, mc2, _, _ = get_weight_mem_configs(1, e, H, N, s01, s2, dram, has_bias=False)
        tw01 = ttnn.from_torch(
            prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, e, H, N, s01),
            dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=mc01,
        )
        tw2 = ttnn.from_torch(
            prepare_w2_tensor_for_moe_compute(w2, 1, e, N, H, s2, s01),
            dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=mc2,
        )
        # The reference uses unrounded weights, so BFP4 rounding alone caps the PCC below 1.
        r01 = (w0, w1, w2)
        out_w = auto_output_width_shard_dim(H)
        drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, OUT_H_SHARDS, out_w, H)
        mapping = (torch.arange(E) // LOCAL).int()[None].repeat(2, 1)  # [devices, experts]: expert e on chip e // 64
        tmap = ttnn.from_torch(
            mapping, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=rep, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for t in args.tokens:
            x = torch.randn(1, t, H)
            ids = torch.stack([torch.randperm(E)[:K] for _ in range(t)])
            scores = torch.softmax(torch.randn(t, K), -1)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=ttnn.L1_MEMORY_CONFIG)
            core = ttnn.CoreCoord(drain.x, drain.y)
            tids = ttnn.from_torch(ids[None].int(), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=l1_on_core(core, (t, K)))
            tsc = ttnn.from_torch(scores[None], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=l1_on_core(core, (t, K)))

            def run():
                outs = ttnn.experimental.moe_compute(
                    tx, tids, tsc, tmap, tw01, tw2,
                    layer_id=0, output_height_shard_dim=OUT_H_SHARDS, intermediate_size=N, cluster_axis=0,
                    activation_type=MoEActivationFunction.GELU,
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                )
                for o in outs[:-1]:
                    o.deallocate(True)
                return outs[-1]

            out = run()
            chips = [ttnn.to_torch(d).float().reshape(K, t, H) for d in ttnn.get_device_tensors(out)]
            ref = torch.zeros(K, t, H)
            for ti in range(t):
                for j in range(K):
                    ex = int(ids[ti, j]) % LOCAL
                    g = x[0, ti] @ r01[0][0, ex]
                    u = x[0, ti] @ r01[1][0, ex]
                    h = torch.nn.functional.gelu(g, approximate="tanh") * u
                    ref[j, ti] = h @ r01[2][0, ex]
            owner = (ids.T // LOCAL)  # [k, T]
            for name, r in (("unweighted", ref), ("score-weighted", ref * scores.T[..., None])):
                for d in range(2):
                    m = owner == d
                    pcc = torch.corrcoef(torch.stack([chips[d][m].flatten(), r[m].flatten()]))[0, 1].item()
                    print(f"T={t} chip {d} ({int(m.sum())} pairs) vs {name}: pcc {pcc:.5f}")
            out.deallocate(True)
            print(f"T={t}: moe_compute {timeit(mesh, run, iters=20)} us", flush=True)


if __name__ == "__main__":
    main()
