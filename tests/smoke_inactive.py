"""Inactive decode rows (position -1, zero page-table row) must not touch live KV.

User 0 owns block 0, which is exactly where an unskipped write from the
inactive row would land.

    pixi run python tests/smoke_inactive.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fake_checkpoint import FakeCheckpoint  # noqa: E402
from tests.smoke_consistency import pcc  # noqa: E402
from tt_models.common.mesh import open_n300  # noqa: E402
from tt_models.models.gemma4.model import Gemma4TT  # noqa: E402


def main():
    ckpt = FakeCheckpoint(["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    n = 40
    seq = torch.randint(10, 20000, (n + 3,))
    with open_n300() as mesh:
        model = Gemma4TT(mesh, ckpt, cache_root=Path(__file__).resolve().parents[1] / "cache" / "fake", max_batch=2, num_layers=2)
        tables = model.allocate_kv_cache(2, n + 3)
        model.prefill(seq[:n], 0, tables[:, 0])
        pt_padded = tables.clone()
        pt_padded[:, 1] = tables[:, 0]
        # Two steps where row 1 is padding and points at user 0's blocks.
        for i in range(2):
            model.decode(torch.stack([seq[n + i], seq[0]]), torch.tensor([n + i, -1]), pt_padded)
        got = model.decode(torch.stack([seq[n + 2], seq[0]]), torch.tensor([n + 2, -1]), pt_padded)[0]
        ref = model.prefill(seq[: n + 3], 0, tables[:, 0])
        print(f"user 0 after padded steps: pcc={pcc(got, ref):.5f} top1 {got.argmax().item()} {ref.argmax().item()}")


if __name__ == "__main__":
    main()
