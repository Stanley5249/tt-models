"""Prototype check of the Q4_0 decode gate|up (tt_models.models.gemma4.expert_q4) against BFP4.

    pixi run python scripts/bench_q4_stream.py [--layer 0] [--seeds 3]

One layer's routed experts for one decode token. Both device paths get the same
x and routing; their act tiles are compared with a float64 host computation
from the checkpoint weights (exact), and each path is timed by trace replay.
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from bench_norm import timeit  # noqa: E402

from tt_models.common.mesh import open_n300  # noqa: E402
from tt_models.common.norm import rms_norm  # noqa: E402
from tt_models.common.weights import WeightLoader  # noqa: E402
from tt_models.models.gemma4.checkpoint import Checkpoint  # noqa: E402
from tt_models.models.gemma4.expert_q4 import down_q4_op, gate_up_q4_op, q4_factor, step_tiles  # noqa: E402
from tt_models.models.gemma4.expert_stream import TILE, pad_down  # noqa: E402
from tt_models.models.gemma4.moe import FeedForward  # noqa: E402


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    p = f"layers.{args.layer}."
    with open_n300(trace_region_size=100_000_000) as mesh:
        loader = WeightLoader(mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8"))
        ff = FeedForward(loader, ckpt, cfg, args.layer)
        ff.decode_constants(1)
        st = ff.stream
        groups = -(-st.kx // TILE)
        E, H, I, tp = ff.n_experts, ff.hidden, cfg.moe_intermediate_size, ff.tp
        n, pad = ff.expert_local, ff.expert_pad - ff.expert_local

        def assembled(t):  # [E, H, 2I] (gate | up) -> device order per chip: gate c, pad, up c, pad
            g, u = t[..., :I], t[..., I:]
            z = t.new_zeros(*t.shape[:-1], pad)
            return torch.cat([torch.cat([g[..., c * n : (c + 1) * n], z, u[..., c * n : (c + 1) * n], z], -1) for c in range(tp)], -1)

        gu = ckpt.get(p + "experts.gate_up_proj").transpose(1, 2)  # [E, H, 2I]
        q, s = q4_factor(gu, dim=-2)  # blocks along H
        w_exact = (q * s.repeat_interleave(TILE, -2)).to(torch.float64)  # clamped Q4_0, exact on device
        q_dev = loader.put(f"L{args.layer}.exp_gate_up_q4_p{ff.expert_pad}", lambda: assembled(q)[None], ttnn.bfloat4_b, shard_dim=-1)
        s_dev = loader.put(
            f"L{args.layer}.exp_gate_up_q4s_p{ff.expert_pad}", lambda: step_tiles(assembled(s), groups)[None], ttnn.bfloat16, shard_dim=-1
        )
        gu64 = gu.to(torch.float64)

        dn = ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous()  # [E, I, H]
        qd, sd = q4_factor(dn, dim=-2)  # blocks along I
        dn_exact = (qd * sd.repeat_interleave(TILE, -2)).to(torch.float64)
        kt = st.kt
        qd_dev = loader.put(f"L{args.layer}.exp_down_q4_p{st.down_width}", lambda: pad_down(qd, st.down_width)[None], ttnn.bfloat4_b, shard_dim=-2)

        def down_steps():  # [E, I / 32, W] -> per chip [E, 32, W], rows 0 .. kt - 1
            sp = pad_down(sd, st.down_width).reshape(E, tp, kt, st.down_width)
            return torch.nn.functional.pad(sp, (0, 0, 0, TILE - kt)).reshape(E, tp * TILE, st.down_width)[None]

        sd_dev = loader.put(f"L{args.layer}.exp_down_q4s_p{st.down_width}", down_steps, ttnn.bfloat16, shard_dim=-2)
        dn64 = dn.to(torch.float64)
        scale = ckpt.get(p + "router.per_expert_scale").double()

        def act_host(xh, ids, wts, w):
            out = []
            for e, we in zip(ids.tolist(), wts.tolist()):
                h = xh @ w[e]
                out.append(gelu_tanh(h[:I]) * h[I:] * we * scale[e])
            return torch.stack(out)  # [slots, I]

        def act_dev(act):  # per chip [32, slots * pairs * 32] row 0 -> [slots, I]
            parts = [ttnn.to_torch(t).double()[0, 0, 0].reshape(ff.top_k, -1)[:, :n] for t in ttnn.get_device_tensors(act)]
            return torch.cat(parts, -1)

        def down_host(act_chips, ids, w):  # act per chip [slots, n] -> per chip [H]
            return [sum(a[j] @ w[e, c * n : (c + 1) * n] for j, e in enumerate(ids.tolist())) for c, a in enumerate(act_chips)]

        def down_dev(out):
            return [ttnn.to_torch(t).double()[0, 0, 0, :H] for t in ttnn.get_device_tensors(out)]

        def snr(a, ref):
            return (10 * torch.log10(ref.pow(2).sum() / (a - ref).pow(2).sum())).item()

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            r = ttnn.from_torch(
                torch.randn(1, 1, 1, H) * 3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )
            normed = rms_norm(r, None, ff.eps)
            x = ttnn.mul(normed, ff.pre_ff2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ids, weights = ff._route(normed, 1)
            xh = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).double()[0, 0, 0]
            idh = ttnn.to_torch(ttnn.get_device_tensors(ids)[0])[0, : ff.top_k].long()
            wh = ttnn.to_torch(ttnn.get_device_tensors(weights)[0]).double()[0, : ff.top_k]
            ref = act_host(xh, idh, wh, gu64)
            ref_q4 = act_host(xh, idh, wh, w_exact)

            def bfp4():
                act = st.act_tensor()
                st.gate_up(x, ids, weights, ff.scale_row, ff.exp_gate_up, act)
                return act

            def q4():
                act = st.act_tensor()
                gate_up_q4_op(
                    st.gu_assign, st.gu_cores, x, ids, weights, ff.scale_row, q_dev, s_dev, act,
                    kx=st.kx, slots_per_core=st.gu_slots, pairs=st.pairs, n_scale=st.n_experts,
                )
                return act

            a4, aq = act_dev(bfp4()), act_dev(q4())
            act_q = q4()
            act_chips = [ttnn.to_torch(t).double()[0, 0, 0].reshape(ff.top_k, -1)[:, :n] for t in ttnn.get_device_tensors(act_q)]
            ref_d = down_host(act_chips, idh, dn64)

            def bfp4_down():
                out = st.out_tensor(ttnn.L1_MEMORY_CONFIG)
                st.down(act_q, ids, ff.exp_down, out)
                return out

            def q4_down():
                out = st.out_tensor(ttnn.L1_MEMORY_CONFIG)
                down_q4_op(
                    st.down_assign, st.down_cores, act_q, ids, qd_dev, sd_dev, out,
                    slots=st.slots, kt=kt, cols=st.down_cols, banks=st.banks, bank_cols=st.bank_cols, n_tiles=st.n_tiles,
                )
                return out

            d4, dq = down_dev(bfp4_down()), down_dev(q4_down())
            for c in range(tp):
                print(f"    down chip {c}: SNR vs checkpoint weights: bfp4 {snr(d4[c], ref_d[c]):.1f} dB, q4 {snr(dq[c], ref_d[c]):.1f} dB", flush=True)
            print(
                f"seed {seed}: SNR vs checkpoint weights: bfp4 {snr(a4, ref):.1f} dB, q4 {snr(aq, ref):.1f} dB "
                f"(q4 vs its own exact weights {snr(aq, ref_q4):.1f} dB, host clamp-only {snr(ref_q4, ref):.1f} dB)",
                flush=True,
            )
        runs = {"bfp4": [], "q4": [], "bfp4 down": [], "q4 down": []}
        for _ in range(5):  # alternate, so drifts hit both alike; report the median
            for name, fn in (("bfp4", bfp4), ("q4", q4), ("bfp4 down", bfp4_down), ("q4 down", q4_down)):
                runs[name].append(timeit(mesh, fn, iters=20))
        for name, v in runs.items():
            print(f"{name} gate_up us median {sorted(v)[2]} (runs {v})", flush=True)


if __name__ == "__main__":
    main()
