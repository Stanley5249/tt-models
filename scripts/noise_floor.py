"""Prefill KL vs fp32 for 512 tokens under tiny embedding perturbations (noise floor).

    pixi run [env PYTHONPATH=<code root>] python scripts/noise_floor.py <label> <seed> [<seed> ...]

A single 512-token run moves by about +-0.04 KL when the input moves by one
BF16 ulp (top-8 routing flips propagate through attention), so compare
means over a few seeds, not single runs.
Seed 0 means no perturbation.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from precision_study import score, tokens  # noqa: E402

from gemma4_tt.checkpoint import Checkpoint  # noqa: E402
from gemma4_tt.mesh import open_n300  # noqa: E402
from gemma4_tt.tt.model import Gemma4TT  # noqa: E402

N = 512
label, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
ckpt = Checkpoint()
ids = tokens(ckpt, N)
base = torch.load(ROOT / "cache" / "ref" / f"readme-{N}-fp32-logits.pt")
with open_n300() as mesh:
    model = Gemma4TT(mesh, ckpt, cache_root=ROOT / "cache", max_batch=1)
    blocks = -(-(N + 64) // model.block_size)
    model.allocate_kv_cache(blocks)
    plain = model.embed
    for seed in seeds:
        if seed:
            g = torch.Generator().manual_seed(seed)

            def embed(t, g=g):
                x = plain(t)
                return (x.float() * (1 + 2**-9 * torch.randn(x.shape, generator=g))).to(x.dtype)

            model.embed = embed
        else:
            model.embed = plain
        logits = model.prefill(ids, torch.arange(blocks, dtype=torch.int32), slot=0, all_logits=True)
        print("RESULT", json.dumps({"label": label, "seed": seed, **score(logits, base, ids)}), flush=True)
