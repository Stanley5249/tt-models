"""Device check of the compact boundary ops (tt_models.models.gemma4.boundary) against torch.

    pixi run python tests/boundary_check.py

Random vectors; per-chip inputs differ so the chip sum is exercised. Reports
max abs error and PCC per output, and times each op by trace replay.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_norm import timeit  # noqa: E402

from tt_models.common.ccl import _SEMAPHORES, init_ccl  # noqa: E402
from tt_models.common.mesh import open_n300  # noqa: E402
from tt_models.models.gemma4.boundary import Boundary  # noqa: E402
from tt_models.models.gemma4.compact import compact_tensor, from_compact, to_compact  # noqa: E402

H, EPS = 2816, 1e-6


def rms(v):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)


def report(name, got, ref):
    got, ref = got.float().flatten(), ref.float().flatten()
    pcc = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    print(f"{name}: max err {(got - ref).abs().max():.5f} (ref max {ref.abs().max():.3f}), pcc {pcc:.6f}", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-cores", type=int, default=11)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4], help="decode tokens per boundary")
    args = ap.parse_args()
    v = torch.randn(2, H)
    assert torch.equal(from_compact(to_compact(v), 2, H), v), "host compact round trip"
    print("host round trip ok")
    torch.manual_seed(0)
    with open_n300(trace_region_size=50_000_000) as mesh:
        init_ccl(mesh)
        b = Boundary(mesh, H, EPS, tail_cores=args.tail_cores)
        shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)
        cores = [ttnn.CoreCoord(i, 7) for i in range(8)]

        def dev(t, dtype=ttnn.float32, mapper=None, mem=ttnn.L1_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(mesh), memory_config=mem)

        def chip(t, i=0):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[i]).float()

        def gather(t):
            s, br = _SEMAPHORES[id(mesh)].next()
            return ttnn.experimental.all_gather_async(
                t, persistent_output_buffer=None, dim=1, multi_device_global_semaphore=s, num_links=1, cluster_axis=1,
                topology=ttnn.Topology.Linear, memory_config=ttnn.L1_MEMORY_CONFIG, barrier_semaphore=br, chunks_per_sync=10,
                num_workers_per_link=2, num_buffers_per_channel=2,
            )

        for R in args.rows:
            print(f"== {R} row(s)", flush=True)
            # pack: per-chip row-form FP32 rows -> compact vectors, after one vector of offset
            rows = torch.randn(2, 1, R, H)
            src = dev(rows, mapper=shard0)
            dst = b.compact(R + 1)
            b.pack(src, dst, cores, dst_chunk0=b.V * 64, rows=R)
            for i in range(2):
                report(f"pack chip {i}", from_compact(chip(dst, i)[0, 0], R + 1, H)[1:], rows[i, 0])
            print("pack us", timeit(mesh, lambda: b.pack(src, b.compact(R), cores, rows=R)))

            # all-gather of compact vectors
            parts = torch.randn(2, R, H)
            pc = dev(torch.stack([to_compact(parts[i]) for i in range(2)])[:, None], mapper=shard0)
            g = gather(pc)
            print("gathered", g.shape, "us", timeit(mesh, lambda: gather(pc)))

            # attention tail
            x = torch.randn(R, H) * 4
            w_attn = torch.rand(3, H) + 0.5
            xt = compact_tensor(mesh, x, ttnn.L1_MEMORY_CONFIG)
            wt = compact_tensor(mesh, w_attn)
            x_out, n, m_in, e_in = b.compact(R), b.row(R), b.row(R), b.row(R)
            b.attn_tail(g, xt, wt, x_out, n, m_in, e_in, rows=R)
            x2 = x + rms(parts.sum(0)) * w_attn[0]
            nn = rms(x2)
            report("attn x2", from_compact(chip(x_out)[0, 0], R, H), x2)
            report("attn n", chip(n)[0, 0, :R], nn)
            report("attn m_in", chip(m_in)[0, 0, :R], nn * w_attn[1])
            report("attn e_in", chip(e_in, 1)[0, 0, :R], nn * w_attn[2])
            print("attn_tail us", timeit(mesh, lambda: b.attn_tail(g, xt, wt, b.compact(R), b.row(R), b.row(R), b.row(R), rows=R)))

            # feed-forward tail: per chip [mlp rows | expert rows]
            ffp = torch.randn(2, 2, R, H)  # [chip, (mlp, experts), row, H]
            fc = dev(torch.stack([to_compact(ffp[i].reshape(2 * R, H)) for i in range(2)])[:, None], mapper=shard0)
            gf = gather(fc)
            w_ff = torch.rand(4, H) + 0.5
            wft = compact_tensor(mesh, w_ff)
            x3, h = b.compact(R), b.row(R)
            scalar = 0.75
            b.ff_tail(gf, x_out, wft, x3, h, scalar, rows=R)
            m, e = ffp[:, 0].sum(0), ffp[:, 1].sum(0)
            t = rms(m) * w_ff[0] + rms(e) * w_ff[1]
            x2d = from_compact(chip(x_out)[0, 0], R, H)
            ref_x3 = (x2d + rms(t) * w_ff[2]) * scalar
            report("ff x3", from_compact(chip(x3)[0, 0], R, H), ref_x3)
            report("ff h", chip(h, 1)[0, 0, :R], rms(ref_x3) * w_ff[3])
            print("ff_tail us", timeit(mesh, lambda: b.ff_tail(gf, x_out, wft, b.compact(R), b.row(R), scalar, rows=R)))


if __name__ == "__main__":
    main()
