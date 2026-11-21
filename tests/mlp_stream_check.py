"""Device check of the dense MLP stream (expert_stream.MlpStream) against the ttnn MLP path.

    pixi run python tests/mlp_stream_check.py [--layer 0]
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_norm import timeit  # noqa: E402

from tt_models.common.mesh import open_n300  # noqa: E402
from tt_models.common.weights import WeightLoader  # noqa: E402
from tt_models.models.gemma4.boundary import Boundary  # noqa: E402
from tt_models.models.gemma4.checkpoint import Checkpoint  # noqa: E402
from tt_models.models.gemma4.compact import from_compact  # noqa: E402
from tt_models.models.gemma4.moe import FeedForward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    ckpt = Checkpoint()
    cfg = ckpt.config
    with open_n300(trace_region_size=50_000_000) as mesh:
        ff = FeedForward(WeightLoader(mesh, Path("cache/gemma4-26b-qat-tp2-bfp8-bfp8-bfp4-bfp8")), ckpt, cfg, args.layer)
        b = Boundary(mesh, cfg.hidden_size, cfg.rms_norm_eps)
        ms = ff.mlp_stream
        torch.manual_seed(0)
        x = ttnn.from_torch(torch.randn(1, 1, 1, cfg.hidden_size), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.L1_MEMORY_CONFIG)
        ref = ff._mlp(x, dtype=ttnn.float32)
        h = ms.h_tensor()
        ms.gate_up(x, ff.mlp_gate_up, h)
        out = b.compact(2)
        ms.down(h, ff.mlp_down, out, compact_chunk0=0)
        for c in range(2):
            r = ttnn.to_torch(ttnn.get_device_tensors(ref)[c]).float()[0, 0, 0]
            o = from_compact(ttnn.to_torch(ttnn.get_device_tensors(out)[c]).float()[0, 0], 2, cfg.hidden_size)[0]
            pcc = torch.corrcoef(torch.stack([o, r]))[0, 1].item()
            print(f"chip {c}: pcc {pcc:.6f}, max diff {(o - r).abs().max():.4f}, ref max {r.abs().max():.3f}", flush=True)
        print("ttnn mlp us", timeit(mesh, lambda: ff._mlp(x, dtype=ttnn.float32)))

        def stream():
            hh = ms.h_tensor()
            ms.gate_up(x, ff.mlp_gate_up, hh)
            o = b.compact(2)
            ms.down(hh, ff.mlp_down, o, compact_chunk0=0)
            hh.deallocate(True)
            return o

        print("stream mlp us", timeit(mesh, stream))
        print("gate_up us", timeit(mesh, lambda: ms.gate_up(x, ff.mlp_gate_up, ms.h_tensor())))


if __name__ == "__main__":
    main()
