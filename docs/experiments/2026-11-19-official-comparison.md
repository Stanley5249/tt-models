# Comparison with the official Tenstorrent results

**Question.** How does single-user serving on one N300d compare with
Tenstorrent's published Gemma 4 26B-A4B numbers, and what can be concluded?

**Setup.** `a527dad` (model code as `f158df1`), vLLM 0.31, vllm-tt-plugin
0.1.0. `scripts/serve/bench_tis.sh` builds requests like
tt-inference-server's release client (chat endpoint, random prompts,
ignore-EOS, greedy), 3 runs after a discarded warm-up. Official sources read
on 2026-11-19 (tt-metal `bbba07b2874`, tt-inference-server `0ad0f4be90`).
Bytes per token from `weight_traffic.py` (last at `7f72451`).

**Result.** 55.0 tok/s/user at 128/128 and one user (TPOT 18.17 ms), against
the official QB2 gate of 45.69 on four Blackhole chips. Newer QB2
configurations reach 53.5 (dedicated code) and 62.7 (with MTP). TTFT was
3-5x worse than QB2 and four users reached only 63.2 tok/s total. This
configuration reads about half the weight bytes per token (BFP4 experts,
BFP8 dense) and is less exact than the official BFP8/BF16 precision.

**Decision.** Only a narrow claim holds: on one N300d at this precision the
single-user speed is in the range of the 4-chip QB2 builds. Hardware,
precision and serving features differ, so no claim of faster or more
efficient code. Follow-ups taken: multi-user decode, prefill, and a GPQA run
([GPQA baseline](2026-11-21-gpqa-baseline.md)).

**Data.** `2026-11-19-official-comparison/`: `report.md` (the full
comparison with labels for measured, published and estimated numbers),
`tis-match/` (vLLM bench JSON), `standalone/`, `official/` (the specs and
references read), `weight-traffic.txt`.
