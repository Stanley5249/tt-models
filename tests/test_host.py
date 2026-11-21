"""Host-only checks of the pure helpers: layouts, padding and routing plans."""

import torch

from tt_models.models.gemma4.compact import compact_tiles, from_compact, to_compact
from tt_models.models.gemma4.config import PREFILL_BUCKET, SHORT_PIECE, prefill_rows
from tt_models.models.gemma4.expert_plan import CHUNK_GRANULE, plan_chunks
from tt_models.models.gemma4.kv_cache import kv_groups
from tt_models.models.gemma4.q4 import factor_blocks


def test_compact_round_trip():
    v = torch.randn(4, 2816)
    assert compact_tiles(2816) == 3
    assert torch.equal(from_compact(to_compact(v), 4, 2816), v)


def test_prefill_rows():
    assert prefill_rows(1, 0) == 32
    assert prefill_rows(SHORT_PIECE, 0) == SHORT_PIECE
    assert prefill_rows(SHORT_PIECE + 1, 0) == 640
    assert prefill_rows(33, 128) == PREFILL_BUCKET  # a later piece always pads to the bucket


def test_kv_groups_gemma4():
    # 25 sliding and 5 global layers: 5 sliding groups of 5 and 1 global group, as vLLM forms them.
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
    groups = kv_groups(types)
    assert sorted(set(groups)) == list(range(6))
    assert all(groups[i] == 5 for i, t in enumerate(types) if t == "full_attention")
    assert all(groups.count(g) == 5 for g in range(6))


def test_plan_chunks_places_every_pair():
    torch.manual_seed(0)
    tokens, k, rows, experts = 50, 8, 32, 128
    ids = torch.stack([torch.randperm(experts)[:k] for _ in range(tokens)])
    w = torch.rand(tokens, k)
    gather, row_w, chunk_expert, row_of_pair = plan_chunks(ids, w, rows, experts)  # row_of_pair is slot-major
    assert chunk_expert.numel() % CHUNK_GRANULE == 0
    assert gather.numel() == row_w.numel() == chunk_expert.numel() * rows
    for pair, row in enumerate(row_of_pair.tolist()):
        j, t = divmod(pair, tokens)
        assert gather[row] == t
        assert row_w[row] == w[t, j]
        assert chunk_expert[row // rows] == ids[t, j]
    padding = torch.ones(gather.numel(), dtype=torch.bool)
    padding[row_of_pair.long()] = False
    assert torch.all(row_w[padding] == 0)


def test_q4_factor_recovers_the_grid():
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (64, 32)).float()
    s = torch.rand(64, 1) * 0.01 + 0.001
    b = (q * s).to(torch.bfloat16).float()
    q2, s2 = factor_blocks(b)
    assert torch.allclose(q2 * s2, b, rtol=0.01, atol=1e-6)
    zero_q, zero_s = factor_blocks(torch.zeros(1, 32))
    assert zero_s.item() == 0 and torch.all(zero_q == 0)
