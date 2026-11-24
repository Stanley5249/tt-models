# Gemma 4 decode reassessment on Wormhole N300d

**Setup.** 2026-11-25, baseline `f87908f` (code from `f979992`), CPU governor `performance`, one N300d and the locked tt-scope v0.5.0 stable stack. The installed vLLM uses one Torch thread. The model and benchmark tokenizers were loaded with `HF_HUB_OFFLINE=1`. Each serving point used one discarded warm-up and seeds 0, 1, and 2.

```sh
HF_HUB_OFFLINE=1 just serve-gemma4
HF_HUB_OFFLINE=1 pixi run python scripts/serve/bench_tis.py session1125-base \
    c1-128-128-n8 c4-128-128-n16 c8-128-128-n32 c16-128-128-n64 \
    c1-2048-128-n4 c16-1024-256-n32 c16-128-128-n32-sampled
HF_HUB_OFFLINE=1 pixi run python scripts/serve/serve_check.py
```

The random dataset chooses distinct vocabulary offsets within a pass. Identical seeds and input lengths across points can reuse the first requests' cached prefixes, so point order matters. The natural-text isolation check separately exercises sixteen concurrent requests.

| Users | Input/output | Mode | Initial tok/s | Optimized tok/s | Initial TPOT (ms) | Optimized TPOT (ms) |
| --: | :-- | :-- | --: | --: | --: | --: |
| 1 | 128/128 | Greedy | 54.90 | 55.96 | 16.50 | 16.15 |
| 4 | 128/128 | Greedy | 144.36 | 151.05 | 23.67 | 22.42 |
| 8 | 128/128 | Greedy | 190.94 | 225.34 | 32.83 | 29.15 |
| 16 | 128/128 | Greedy | 288.11 | 317.73 | 44.42 | 39.24 |
| 1 | 2048/128 | Greedy | 36.98 | 37.40 | 17.29 | 16.99 |
| 16 | 1024/256 | Greedy | 170.55 | 181.33 | 74.55 | 69.05 |
| 16 | 128/128 | Sampled | 257.10 | 255.54 | 50.05 | 50.28 |

Throughput includes prefill. At one user and 128/128, the baseline decode rate rose from 60.6 to 61.9 tokens/s from mean TPOT. At eight users, all runs completed. One reached 150.23 tokens/s versus 210.85 and 211.73 for the others. Its P90 TTFT was 8.06 s versus about 1.02 s while median ITL stayed at 31.31 ms versus 31.39 ms. During its request window, 67 kernel include files were added to the local JIT cache, including matmul, fill/pad and gather kernels. This connects the outlier with cold program compilation and the result stays in the raw mean.

**LM-head output page size.** Kept logits bit-identical and compared one FP32 page per vocabulary row with 4,096, 1,024, 256, and 32 elements per page. At 16 rows, the original took 1.924 ms for blocking head replay and 3.683 ms for readback. The finer layouts did not improve the combined time. With 32 elements per page the read rose to 7.687 ms. At four rows, the best alternative (1,024 elements) saved 0.22 ms across head and readback. The layout alone was not sufficient to justify the change.

**Greedy scan on the host.** On one thread, paired CPU tests with three input seeds measured Torch/NumPy at 0.150/0.033 ms for one row, 0.619/0.092 ms for four, 1.238/0.182 ms for eight, and 2.518/0.423 ms for sixteen. NumPy kept the first vocabulary index for ties, signed zero, and NaNs.

**Tracy and memory.** The locked tt-scope CLI captured six layers at batch 1 and 16 with `tracy --programs 20000`. Both captures passed marker and clock checks. The single-row replay spanned 4.326 and 4.296 ms on chips 0 and 1. The 16-row replay spanned 8.231 and 8.161 ms. The device clocks were read separately. At 16 rows, mean layer time on chip 0 included 511 µs for expert gate/up and down, 114 µs for the two tails, 93 µs for the two gathers, and 73 µs for QKV. At one row, QKV and the expert pair each took about 72 µs per layer. The LM head took 1.671 ms at one row and 1.754 ms at sixteen in the captures. The head streams 393 MB of BFP8 weights per chip, for a 1.36 ms DRAM-only lower bound at 288 GB/s per chip. [N300d specifications](https://docs.tenstorrent.com/aibs/wormhole/index.html)

The six-layer tt-scope memory report passed with 1,559 operations and 419,681 buffer records. The 16-row gather NoC capture passed on both chips. For a replay of a 196,608-byte input per chip, the on-chip trace recorded 393,728 bytes of reads and 597,248 bytes of writes on each chip, including forwarding and staging. The NoC counts describe local traffic. The single-link timing measurements use plain trace replay.

**Retained changes.**

- `bf41a37` reduces FP32 logits on each chip for all-greedy batches with no logprobs. The host receives one candidate per worker and row, and merges in vocabulary order. The prototype took 1.149 ms for the device scan and 0.547 ms for the compact read and merge at 16 rows, versus 3.683 ms for reading full logits plus 2.518 ms for Torch argmax. The NumPy host experiment provided a lower-cost comparison. Device greedy made that host change redundant.
- `9d2b250` shares identical RoPE tiles. The input now holds 11 FP32 tiles per row including the residual, down from 21. At 16 rows that is about 704 KiB versus 1,344 KiB per replicated input. The 30-layer raw-logit replay, mean over three distinct-token seeds, moved from 37.109 to 36.693 ms at 128 tokens and from 41.672 to 41.293 ms at 1,024. The compared FP32 logit digests matched in all 15 seed/workload pairs.
- `e87dbd9` uses one fabric worker for small compact gathers. The 3- and 6-tile gathers dropped from 10.94 to 7.97 µs and from 12.03 to 10.28 µs. The larger payloads kept two workers.

**Other probes.** Group-major LM-head weights gave the same FP32 logits. Head replay at 16 rows rose from 1.834 ms to 1.855–1.946 ms across packet sizes, and the best one-row improvement was 0.036 ms. Logit page sizing alone was also rejected. L1 SDPA output and a custom concatenation op produced matching digests in six-layer replay. The direct concatenation saved 0.50 ms at 11 rows, where the original uses a permute fallback. It saved 0.05 ms at 16 rows and was slower at one row, so the generic replacement was left out. Restricting the router's top-k reads to selected rows changed the isolated 16-row router from 68.4 to 64.8 µs, about 0.11 ms per 30-layer step if fully exposed. That result alone was too small to justify changing the routing kernel.

**Serving and correctness.** The optimized checkout completed all serving requests and the default `serve_check.py` suite. Both rounds of the sixteen-way natural-text isolation check matched the isolated runs token for token. The sampled, prefix-hit, cancel, long-arrival and tool-call checks passed. Across the paired synthetic greedy benchmarks, 467 of 468 visible completions matched. One 16-user long-prompt result had 13 visible characters where the baseline had none, with 256 generated tokens on both runs. The device suite passed 10 tests, including same-shape argmax calls on separate buffers and reuse of the same trace for greedy and raw logits.

The same-day FP32 reference comparison used `scripts/accuracy/compare.py --decode-steps 32 --offsets 0 2048 --trace --batch B --perturb SEED` for `B = 1, 16` and `SEED = 0, 1, 2`. Reported per-position metrics matched for all 192 decode positions per batch. Mean KL was 0.30017 at one row and 0.31062 at 16 rows. Top-1 agreement was 87.50% and 85.94%, respectively. The 16-row reference check used repeated input rows and scored row 0. The separate fixed-token replay and isolation checks exercised distinct inputs.

**Stopping condition.** Five further rounds followed the retained changes. Each started from the validated checkout or used the same input and existing configuration as its control.

| Round | Measured result | Decision |
| :-- | :-- | :-- |
| TTNN subcore grid for uneven batches | Six layers at 9, 11, and 15 rows changed from 9.680/10.873/13.004 to 9.619/10.922/13.018 ms. The logit digests matched. | Kept the existing concat path. |
| 256-position SDPA chunks with aligned page tables | Six layers at one row and 8,192 positions went from 5.193 to 5.131 ms. At 16 rows and 1,024 positions the replay rose from 14.033 to 14.139 ms. Logit digests changed. | Kept 128-position chunks where the page table allows them. |
| Skip building unchanged host page tensors | Full-model replay changed from 15.255 to 15.222 ms at 1 row, 20.699 to 20.619 ms at 4, and 41.391 to 41.403 ms at 16 rows and 1,024 positions. Logit digests matched. | Kept the existing preparation. |
| Three or five expert gate/up readers per bank | With 84–86 distinct experts, gate/up took 697/645/687 µs with 3/4/5 readers. At one row, five tied four at 49.9 µs. | Kept four. |
| Broadcast fabric gather | Saving 1.31 µs on the four-row gather came with a 3.03 µs slowdown for 16 rows and a small one-row penalty. | Kept the existing gather. |

Full-vocabulary sampled readback and routed-expert DRAM traffic remain the largest opportunities at high concurrency. A single-user device optimization has to improve QKV, the boundary operations, or the LM head enough to beat the measured spread. The shorter greedy readback requires no extra trace or quantization.

**Data.** [Serving repetitions](2026-11-25-gemma4-decode-readback/serving.json) and [FP32 reference repetitions](2026-11-25-gemma4-decode-readback/reference.json). Raw benchmark outputs and logs are in `artifacts/serve/session1125-base`, `artifacts/serve/session1125-final`, and `artifacts/session-1125`. The four fresh tt-scope captures under `artifacts/runs` have names ending in `session1125-b1`, `session1125-b16`, `session1125-memory`, and `session1125-noc`.
