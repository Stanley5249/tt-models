# Shared expert reads and decode up to 16 rows

**Question.** A decode step read every row's 8 routed experts separately (slot = 8 x row + j), so rows that picked the same expert read it twice, and the expert scratch capped decode at 4 rows. Does reading each distinct expert once pay off, and how far can the row count go?

**Setup.** Baseline `c9933b5`, change `0499452` plus the Q4_0 routing fix `perf/union-experts`, measured on 2026-11-23, CPU governor `performance`. The router op lists the step's distinct experts (ids with their count, and a slot-major weight per row that is zero for rows that did not pick the expert); the stream kernels read the count at runtime. `STREAM_ROWS` goes from 4 to 16 (router expert-tile split, fewer tail cores per row past 10 rows, a permute fallback for concatenating heads at 9-15 rows).

- Standalone: `scripts/bench/decode.py --batch B --steps 128 --trace` (64 steps at 8 and 16 rows), which now prefills every row with its own random prompt (`3292aa2`); the baseline ran the same script with `PYTHONPATH` on a `c9933b5` worktree. Its time includes the host softcap of every row (about 0.34 ms per row), which the vLLM path skips.
- Serving: `just serve-gemma4 --max-num-seqs <4 | 16>`, then `scripts/serve/bench_tis.py <label> c1-128-128-n8 c4-128-128-n16 c8-128-128-n32 c16-128-128-n64`, 3 repetitions after a warm-up.
- Accuracy: `scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --batch B [--perturb k]`, k = 0..2.
- Kernels: `scripts/kernels/router_check.py --rows 1 2 4 8 16`, `scripts/kernels/expert_stream_check.py --rows 1 4 16`, `just test -m device` with 11 and 16 rows added.

**Result.** Standalone decode, median ms per step (total tok/s):

| Rows |      Baseline |  Shared reads |
| ---: | ------------: | ------------: |
|    1 |  16.94 (59.0) |  16.94 (59.0) |
|    2 |  21.71 (92.1) |  21.16 (94.5) |
|    4 | 28.81 (138.9) | 26.01 (153.8) |
|    8 |             - | 37.70 (212.2) |
|   16 |             - | 55.88 (286.3) |

Random prompts give 29-30 distinct experts for 4 rows and 84-86 for 16 (`expert_stream_check.py`); real text should share more. At 16 rows the expert gate|up takes 637 us and the down 267 us per layer, and the router 145 us (167 us before the expert-tile split), most of it in the serial top-k on one RISC.

Serving, 128 in / 128 out, mean of 3 repetitions:

| Users |          Baseline TPOT | Baseline tok/s | New TPOT | New tok/s | New TTFT |
| ----: | ---------------------: | -------------: | -------: | --------: | -------: |
|     1 |               17.50 ms |           52.3 | 17.56 ms |      52.1 |   226 ms |
|     4 |               29.63 ms |          115.4 | 27.08 ms |     124.6 |   668 ms |
|     8 | 30.75 ms (4 at a time) |          115.7 | 35.36 ms |     172.5 |   1.44 s |
|    16 | 30.08 ms (4 at a time) |          115.7 | 51.54 ms |     216.2 |   2.92 s |

The baseline server takes 4 requests at a time, so its throughput stops at 115.7 tok/s. At 16 users TTFT grows because prefills run one request at a time between decode steps.

Accuracy, mean of 3 seeds (identical rows, row 0 scored): batch 1 and batch 4 give the same KL as the baseline to four digits (0.4436 and 0.4338), because identical rows share their 8 experts in top-k order. Batch 16 gives KL 0.4317, top-1 0.760. The router matches an fp32 reference within 0.0007 at every row count; the 16-row stream check's one low row (PCC 0.91) was a near-tie (logits 0.8025 against 0.8001) that the router op resolved the fp32 way and the prefill path did not.

**Decision.** Kept; `serve-gemma4` now sets `--max-num-seqs 16`. Single-user TPOT moved 17.50 -> 17.56 ms across 3 repetitions each, a 0.3% cost of the runtime slot count. Reviewed by Sol 6.2 (medium): the Q4_0 prefill path still routed through `topk_router_gpt`, whose output has no count entry; fixed. Skipped: returning early on cores with an empty run, which would delay the activation reads of the busy cores.

**Data.** `2026-11-23-shared-expert-reads/`: `decode/` (r1-base and r1-new are the A/B runs, r2 the 8- and 16-row runs), `accuracy/`, `serve/<label>/` (bench results and `summary.txt`).
