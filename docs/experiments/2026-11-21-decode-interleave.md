# Decode steps between prefill chunks

**Question.** How many prefill chunks should run per decode step, and how large, so that a long prompt arriving during a stream stalls the stream least without costing its own TTFT?

**Setup.** `1941c2f`. An 8,192-token prompt arrives during a running stream (`scripts/serve/serve_check.py arrival` plus server logs with `GEMMA4_TIMING=1`), steady state, 3 runs each.

**Result.**

| Chunk, prefill steps per decode step | Longest stream gap | 8K TTFT |
| ------------------------------------ | -----------------: | ------: |
| 2,048, 2 (plugin default)            |             3.38 s |  6.53 s |
| 2,048, 1                             |             1.72 s |  6.56 s |
| 1,024, 1                             |             1.00 s |  7.18 s |

Before chunked prefill the same arrival held the stream for 5.66 s.

**Decision.** 2,048-token chunks with one decode step after each (`decode_interleave_prefill_steps: 1` in `serve-gemma4`): half the gap at the same TTFT. 1,024-token chunks would cut the gap further at 10% more TTFT.

**Data.** None; the numbers are in the commit message of `1941c2f`.
