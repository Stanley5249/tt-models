"""Device smoke test with random weights: decode(L) must match prefill(L+1).

    pixi run python tests/smoke_consistency.py --prompt-len 64
    pixi run python tests/smoke_consistency.py --prompt-len 1100   # exercises the sliding ring
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma4_tt.mesh import open_n300  # noqa: E402
from gemma4_tt.tt.model import Gemma4TT  # noqa: E402
from tests.fake_checkpoint import FakeCheckpoint  # noqa: E402


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    ckpt = FakeCheckpoint(["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    ids = torch.randint(10, 20000, (args.prompt_len + args.steps,))
    cache_root = Path(__file__).resolve().parents[1] / "cache" / "fake"
    with open_n300() as mesh:
        t0 = time.time()
        model = Gemma4TT(mesh, ckpt, cache_root=cache_root, max_batch=1, num_layers=args.layers)
        print(f"load {time.time() - t0:.0f}s", flush=True)
        blocks = -(-(ids.shape[0] + 64) // model.block_size)
        model.allocate_kv_cache(blocks)
        pt = torch.arange(blocks, dtype=torch.int32)
        n = args.prompt_len
        model.prefill(ids[:n], pt, slot=0)
        decoded = []
        for i in range(args.steps):
            t0 = time.time()
            decoded.append(model.decode(ids[n + i : n + i + 1], torch.tensor([n + i]), pt[None])[0])
            print(f"decode step {i}: {time.time() - t0:.2f}s", flush=True)
        for i in range(args.steps):
            ref = model.prefill(ids[: n + i + 1], pt, slot=0)
            print(f"pos {n + i}: decode vs prefill pcc={pcc(decoded[i], ref):.5f} top1 {decoded[i].argmax().item()} {ref.argmax().item()}")


if __name__ == "__main__":
    main()
