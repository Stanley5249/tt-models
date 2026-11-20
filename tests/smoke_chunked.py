"""Device smoke test with random weights: prefill in chunks and from a shared prefix.

Each case must match one prefill of the whole sequence, and a decode step after
the chunks must match prefill(L + 1):

- chunked: [0, a), [a, b), [b, L) on the CHUNK_ALIGN grid, past the sliding window;
- odd start: a chunk that starts on a 64-token block but off the 128 grid, which
  the model recomputes from the grid without rewriting the shared sliding block;
- prefix hit: a second sequence shares the first P tokens and the first one's
  blocks for them, as vLLM's prefix cache hands them out, and computes from P.

    pixi run python tests/smoke_chunked.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma4_tt.mesh import open_n300  # noqa: E402
from gemma4_tt.tt.model import Gemma4TT  # noqa: E402
from tests.fake_checkpoint import FakeCheckpoint  # noqa: E402


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def report(name, got, ref):
    ok = pcc(got, ref) > 0.9995 and got.argmax() == ref.argmax()
    print(f"{'PASS' if ok else 'FAIL'} {name}: pcc={pcc(got, ref):.5f} top1 {got.argmax().item()} {ref.argmax().item()}", flush=True)
    return ok


def main():
    ckpt = FakeCheckpoint(["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    L, P = 2900, 1536
    seq = torch.randint(10, 20000, (L + 1,))
    other = torch.cat([seq[:P], torch.randint(10, 20000, (L - P,))])
    cache_root = Path(__file__).resolve().parents[1] / "cache" / "fake"
    ok = True
    with open_n300() as mesh:
        model = Gemma4TT(mesh, ckpt, cache_root=cache_root, max_batch=1, num_layers=2)
        tables = model.allocate_kv_cache(4, L + 1)

        ref = model.prefill(seq[:L], 0, tables[:, 0])
        ref_next = model.prefill(seq[: L + 1], 0, tables[:, 0])

        t = tables[:, 1]
        for a, b in ((0, 1024), (1024, 2048)):
            model.prefill(seq[:b], a, t)
        got = model.prefill(seq[:L], 2048, t)
        ok &= report("chunked [0,1024) [1024,2048) [2048,L)", got, ref)
        step = model.decode(seq[L : L + 1], torch.tensor([L]), t[:, None])[0]
        ok &= report("decode after chunked prefill", step, ref_next)

        t = tables[:, 2]
        model.prefill(seq[:1088], 0, t)
        got = model.prefill(seq[:L], 1088, t)  # 1088 = 17 sliding blocks, not on the 128 grid
        ok &= report("odd start 1088", got, ref)

        # Row 3 reads row 0's blocks below P, as a prefix-cache hit would.
        t = tables[:, 3].clone()
        for g, block in enumerate(model.kv.block_sizes):
            t[g, : P // block] = tables[g, 0, : P // block]
        hit = model.prefill(other[:L], P, t)
        fresh = model.prefill(other[:L], 0, tables[:, 1])
        ok &= report(f"prefix hit at {P}", hit, fresh)
        # Row 0's decode reads the shared blocks through its global layer.
        again = model.decode(seq[L : L + 1], torch.tensor([L]), tables[:, :1])[0]
        ok &= report("row 0 decode after the hit (shared blocks intact)", again, ref_next)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
