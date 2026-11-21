# Serving backend: end-to-end numbers

**Question.** After the hybrid KV cache, trace fixes and batched streams (`feat/serving`, merged at `95c8036`), how does serving compare with round 2, and what did the follow-up milestones change?

**Setup.** `scripts/serve/bench_tis.sh` (tt-inference-server's client recipe), mean of 3 runs after a discarded warm-up, CPU governor `performance`. Round 2 from [official comparison](2026-11-19-official-comparison.md); "now" is `feat/quality-sampling` (`16b7a3b`).

**Result.**

| Point                    | TTFT round 2 / serving / now | TPOT round 2 / serving / now |
| ------------------------ | ---------------------------: | ---------------------------: |
| 1 user, 128 in / 128 out |           237 / 238 / 236 ms |     18.17 / 18.45 / 18.08 ms |
| 1 user, 2,048 / 128      |         1.57 / 1.48 / 1.21 s |     19.32 / 19.67 / 19.33 ms |
| 1 user, 8,192 / 128      |         6.95 / 5.15 / 4.36 s |     19.69 / 20.02 / 19.65 ms |
| 4 users, 128 / 128       |           736 / 694 / 695 ms |     57.95 / 37.75 / 36.19 ms |

Single-user TPOT at `feat/serving` was 1.5% slower while standalone decode was unchanged; the likely cause, host bookkeeping for six KV groups instead of one, was not profiled. Sampled requests (temperature 1, top-k 64, top-p 0.95): 18.50 ms at 1 user, 37.34 ms at 4. A cached 6,144-token prefix: 323 ms median TTFT. `serve_check.py`'s 8K follow-up turn: 1.29-1.69 s cached, 7.36-7.76 s with a cache salt.

Not kept, measured along the way: reading logits into a kept host tensor (1.0 ms against 0.66 ms for `to_torch`), two Ethernet links for prefill all-gathers (hung; the N300 fabric runs one link between its chips), and a row-major concat of expert outputs (no change).

**Decision.** Merged. Four users were still slow per user, which led to [compact boundaries for 2-4 rows](2026-11-21-compact-boundaries-multi-row.md).

**Data.** `2026-11-21-serving-backend/feat-serving/` (vLLM bench JSON, prefix runs `c1-prefix6144-*`). The `feat/quality-sampling` runs are in [prefill length buckets](2026-11-21-prefill-length-buckets.md).
