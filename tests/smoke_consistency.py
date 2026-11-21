"""Device smoke test with random weights: decode(L) must match prefill(L+1).

    pixi run python tests/smoke_consistency.py --prompt-len 64
    pixi run python tests/smoke_consistency.py --prompt-len 1100   # past the sliding window
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fake_checkpoint import FakeCheckpoint  # noqa: E402
from tt_models.common.mesh import open_n300  # noqa: E402
from tt_models.models.gemma4.model import Gemma4TT  # noqa: E402


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    ckpt = FakeCheckpoint(["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    lens = [args.prompt_len + 37 * i for i in range(args.batch)]  # different lengths per user
    seqs = [torch.randint(10, 20000, (n + args.steps,)) for n in lens]
    cache_root = Path(__file__).resolve().parents[1] / "cache" / "fake"
    with open_n300() as mesh:
        t0 = time.time()
        model = Gemma4TT(mesh, ckpt, cache_root=cache_root, max_batch=args.batch, num_layers=args.layers)
        print(f"load {time.time() - t0:.0f}s", flush=True)
        tables = model.allocate_kv_cache(args.batch, max(lens) + args.steps)
        for b in range(args.batch):
            model.prefill(seqs[b][: lens[b]], 0, tables[:, b])
        decoded = []
        for i in range(args.steps):
            toks = torch.stack([seqs[b][lens[b] + i] for b in range(args.batch)])
            pos = torch.tensor([lens[b] + i for b in range(args.batch)])
            t0 = time.time()
            decoded.append(model.decode(toks, pos, tables))
            print(f"decode step {i}: {time.time() - t0:.2f}s", flush=True)
        for b in range(args.batch):
            for i in range(args.steps):
                ref = model.prefill(seqs[b][: lens[b] + i + 1], 0, tables[:, b])
                got = decoded[i][b]
                print(f"user {b} pos {lens[b] + i}: decode vs prefill pcc={pcc(got, ref):.5f} top1 {got.argmax().item()} {ref.argmax().item()}")


if __name__ == "__main__":
    main()
