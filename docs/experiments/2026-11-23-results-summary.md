# Results summary by optimization stage

**Question.** Which before/after pairs represent each optimization stage, and how do different runs of the same commit in earlier docs relate to each other?

**Setup.** No new measurements. Every value comes from the linked docs and their data folders. The summary groups the work into three stages:

| Stage | Earlier docs call it | Commits | Dates |
| --- | --- | --- | --- |
| Stage 1: Initial Optimization | round 1 | `5101e0e` -> `5887694` | 2026-11-17 |
| Stage 2: Single-user Decode | round 2 | `5887694` -> `f158df1` | 2026-11-18/19 |
| Stage 3: Multi-user Decode | batched expert streams (step 1), router op and compact boundaries for 2-4 rows (step 2) | `a485d83` -> `0e65f24`, `d565603` -> `333c5e8` | 2026-11-21 |

A stage holds many changes, one per iteration of the profile, change, and benchmark loop. The move from ttnn 0.81.1 to 0.82.0 on 2026-11-22 is a library version change, not a stage.

Metrics, as the docs use them:

- Replay time: the blocking `execute_trace` call on the host, profiler off (`trace_time.py`).
- Standalone decode: the timed `model.decode` call in `scripts/bench/decode.py --trace`: input copy, replay, and logit readback. Sampling is outside the timer.
- TPOT: vLLM's mean time per output token after the first, measured by the client.
- Tokens/s per user: 1000 / TPOT in ms.
- Output throughput: all output tokens over the benchmark's wall time, summed over users. The wall time includes TTFT, so it is below users x tokens/s per user.

**Result.** One pair per stage, each measured back to back on one day:

| Stage | Measure | Before -> after |
| --- | --- | --- |
| 1 | vLLM TPOT, 1 user, 128/128, `bench.sh` with sampling | 74.7 -> 35.2 ms |
| 2 | Replay time, batch 1 | 31.5 -> 15.6 ms |
| 2 | vLLM TPOT, 1 user, 128/128, `bench.sh` with sampling | 35.16 -> 18.72 ms (28.4 -> 53.4 tokens/s per user) |
| 3, step 1 | Standalone decode, batch 4 | 56.49 -> 35.42 ms |
| 3, step 2 | Standalone decode, batch 4 | 34.95 -> 29.49 ms |
| 3, step 2 | vLLM, 4 users, 128/128, `bench_tis.sh` | TPOT 36.17 -> 30.77 ms, output throughput 96.9 -> 111.3 tok/s |

Context for values that differ between docs:

- [Round 1](2026-11-17-decode-round-1/report.md) reports median TPOT (74.8 ms at `5101e0e`). The other docs use the mean, which is 74.71 ms in `baseline-5101e0e/c1-128-128.json`.
- `5887694` was measured twice: 35.18 ms TPOT on 2026-11-17 and 35.16 ms on 2026-11-19 with the CPU governor at `performance`. Its standalone decode was 32.9 and 33.3 ms, inside run-to-run noise.
- Round 2's report commit `a527dad` had overwritten three of round 1's final serving runs. This change moves the 2026-11-19 runs to `2026-11-19-decode-round-2/final-serving/` (18.72 ms TPOT at one user) and restores round 1's 2026-11-17 runs from `d775003` (35.18 ms).
- Round 2's end code gave 18.72 ms with `bench.sh` and sampling, and 18.17 ms with `bench_tis.sh` and greedy decoding ([official comparison](2026-11-19-official-comparison.md)). The client differs, not the code.
- Step 1 ended at 35.42 ms and step 2 started at 34.95 ms. Two merges (`2d0d735`, `d565603`) came in between, and no doc explains the 0.47 ms difference.
- On 2026-11-22, single-user TPOT was 18.10-18.16 ms on ttnn 0.81.1 and 17.46-17.47 ms on 0.82.0 ([ttnn 0.82.0 run](2026-11-22-shared-stack.md)). The 4-user output throughput was not reported that day, so 111.3 tok/s from 2026-11-21 is the latest.

The comparison with Tenstorrent's QB2 holds only at one user with the same client recipe: 57.3 tokens/s per user here (2026-11-22) against the QB2 release gate of 45.69. Hardware, precision, and serving features differ, so the comparison gives no speedup ratio. The 4-user result here and the 32-user QB2 gate run at different concurrency and are context only.

**Decision.** These pairs are the representative numbers per stage. The earlier docs keep their own numbers.

**Data.** None; the values come from the linked docs.
