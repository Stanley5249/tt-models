"""Device checks with random weights (2 layers, real shapes): run with `pytest -m
device`.

Each check compares two device paths that must agree, so the thresholds are
tight: decode against a fresh prefill of the same positions, chunked and
prefix-hit prefill against one full prefill.
"""

from pathlib import Path

import pytest
import torch
import ttnn
from fake_checkpoint import FakeCheckpoint

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.model import Gemma4TT

pytestmark = pytest.mark.device

CACHE = Path(__file__).resolve().parents[1] / "cache" / "fake"
PCC = 0.9995


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[
        0, 1
    ].item()


def assert_close(got, ref, what):
    assert pcc(got, ref) > PCC, f"{what}: pcc {pcc(got, ref):.5f}"
    assert got.argmax() == ref.argmax(), f"{what}: argmax differs"


# one mesh per test: a model keeps its traces and KV until the mesh closes
@pytest.fixture
def mesh():
    with open_n300() as m:
        yield m


def model(mesh, max_batch):
    return Gemma4TT(
        mesh,
        FakeCheckpoint(["sliding_attention", "full_attention"]),
        cache_root=CACHE,
        max_batch=max_batch,
        num_layers=2,
    )


@pytest.mark.parametrize(("prompt_len", "batch"), [(64, 1), (64, 4), (1100, 1)])
def test_decode_matches_prefill(mesh, prompt_len, batch):
    """decode(L) must match prefill(L + 1); 1100 tokens runs past the sliding window."""
    torch.manual_seed(0)
    steps = 2
    lens = [prompt_len + 37 * i for i in range(batch)]  # different lengths per user
    seqs = [torch.randint(10, 20000, (n + steps,)) for n in lens]
    m = model(mesh, batch)
    tables = m.allocate_kv_cache(batch, max(lens) + steps)
    for b in range(batch):
        m.prefill(seqs[b][: lens[b]], 0, tables[:, b])
    decoded = []
    for i in range(steps):
        toks = torch.stack([seqs[b][lens[b] + i] for b in range(batch)])
        decoded.append(m.decode(toks, torch.tensor([n + i for n in lens]), tables))
    for b in range(batch):
        for i in range(steps):
            assert_close(
                decoded[i][b],
                m.prefill(seqs[b][: lens[b] + i + 1], 0, tables[:, b]),
                f"user {b} step {i}",
            )


def scheduled(m, table: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """table [groups, width] as vLLM sends it for computing [start, end)."""
    t = table.clone()
    for g, block in enumerate(m.kv.block_sizes):
        t[g, -(-end // block) :] = 0  # not allocated yet
        if m.kv.sliding[g]:
            t[g, : max(start - m.window + 1, 0) // block] = (
                0  # freed: wholly outside the window
            )
    return t


def test_chunked_prefill_and_prefix_hit(mesh):
    """Chunks, a prefix-cache hit and the null block, with page tables shaped as vLLM
    sends them.
    """
    torch.manual_seed(0)
    L, P = 2900, 1536
    seq = torch.randint(10, 20000, (L + 1,))
    other = torch.cat([seq[:P], torch.randint(10, 20000, (L - P,))])
    m = model(mesh, 1)
    tables = m.allocate_kv_cache(4, L + 1)
    ref = m.prefill(seq[:L], 0, tables[:, 0])
    ref_next = m.prefill(seq[: L + 1], 0, tables[:, 0])

    t = tables[:, 1]
    for a, b in ((0, 1024), (1024, 2048)):
        m.prefill(seq[:b], a, scheduled(m, t, a, b))
    assert_close(
        m.prefill(seq[:L], 2048, scheduled(m, t, 2048, L)), ref, "chunked prefill"
    )
    step = m.decode(
        seq[L : L + 1], torch.tensor([L]), scheduled(m, t, L, L + 1)[:, None]
    )[0]
    assert_close(step, ref_next, "decode after chunked prefill")

    # Row 3 reads row 0's blocks below P, as a prefix-cache hit would.
    t = tables[:, 3].clone()
    for g, block in enumerate(m.kv.block_sizes):
        t[g, : P // block] = tables[g, 0, : P // block]
    hit = m.prefill(other[:L], P, scheduled(m, t, P, L))
    assert_close(hit, m.prefill(other[:L], 0, tables[:, 1]), f"prefix hit at {P}")
    # Row 0's decode reads the shared blocks through its global layer.
    assert_close(
        m.decode(seq[L : L + 1], torch.tensor([L]), tables[:, :1])[0],
        ref_next,
        "row 0 after the hit",
    )

    null = [
        ttnn.to_torch(ttnn.get_device_tensors(c)[0])[0]
        for kv in m.kv.layers
        for c in (kv.k, kv.v)
    ]
    assert all(c.abs().max() == 0 for c in null), "prefill wrote the null block"
    with pytest.raises(ValueError, match=r"must start on the .* block grid"):
        m.prefill(seq[:L], 1088, tables[:, 2])  # off the block grid


def test_inactive_rows_leave_live_kv_alone(mesh):
    """Padding rows (position -1) point at user 0's block 0, where an unskipped write
    would land.
    """
    torch.manual_seed(0)
    n = 40
    seq = torch.randint(10, 20000, (n + 3,))
    m = model(mesh, 2)
    tables = m.allocate_kv_cache(2, n + 3)
    m.prefill(seq[:n], 0, tables[:, 0])
    padded = tables.clone()
    padded[:, 1] = tables[:, 0]
    for i in range(3):
        got = m.decode(
            torch.stack([seq[n + i], seq[0]]), torch.tensor([n + i, -1]), padded
        )[0]
    assert_close(
        got, m.prefill(seq[: n + 3], 0, tables[:, 0]), "user 0 after padded steps"
    )
