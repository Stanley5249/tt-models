# Decode and serving, round 3: 16 users, LM head stream, batched prefill

**Question.** After [shared expert reads](2026-11-23-shared-expert-reads.md) lifted decode to 16 rows, what else limits single-user latency and multi-user throughput, and how far do they move end to end?

**Setup.** Branch `perf/union-experts`, 2026-11-23, CPU governor `performance`. Baseline `c9933b5` (the start of the day), final `f979992`. Each change was measured against its parent commit the same day with `PYTHONPATH` on a worktree of the parent:

    pixi run python scripts/bench/decode.py --batch <B> --steps 64 --trace       # distinct prompt per row
    pixi run python scripts/serve/bench_tis.py <label> c1-128-128-n8 c4-128-128-n16 c8-128-128-n32 c16-128-128-n64 c1-2048-128-n4 c16-1024-256-n32
    pixi run python scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace [--batch B] [--perturb k]   # k = 0..2

Profiles: tt-scope `tracy` captures of `scripts/profile/workload.py --trace --layers 6` at 1 and 16 rows (6 layers, so the device profiler read fits the 30-minute limit), and `py-spy record --subprocesses` of a 16-user server run.

**Result.** Standalone decode, median ms per step (total tok/s):

| Rows |     `c9933b5` |     `f979992` |
| ---: | ------------: | ------------: |
|    1 |  16.97 (58.9) |  15.69 (63.8) |
|    2 |  21.80 (91.7) | 18.19 (109.9) |
|    4 | 28.81 (138.9) | 22.46 (178.1) |
|    8 |             - | 31.06 (257.5) |
|   16 |             - | 46.50 (344.1) |

Serving, mean of 3 repetitions (`c9933b5` serves 4 users at a time, `f979992` 16):

| Point              | `c9933b5` TPOT, tok/s, TTFT | `f979992` TPOT, tok/s, TTFT |
| ------------------ | --------------------------- | --------------------------- |
| 1 user, 128/128    | 17.50 ms, 52.3, 226 ms      | 16.49 ms, 55.0, 235 ms      |
| 4 users, 128/128   | 29.63 ms, 115.4, 670 ms     | 23.65 ms, 145.3, 519 ms     |
| 8 users, 128/128   | 30.75 ms, 115.7, 4.40 s     | 31.28 ms, 211.9, 859 ms     |
| 16 users, 128/128  | 30.08 ms, 115.7, 12.2 s     | 44.23 ms, 289.3, 1.46 s     |
| 1 user, 2048/128   | 18.72 ms, 33.6, 1.44 s      | 17.30 ms, 36.9, 1.27 s      |
| 16 users, 1024/256 | 37.80 ms, 96.6, 25.2 s      | 74.18 ms, 171.2, 4.95 s     |

The changes after shared expert reads, each against its parent (standalone ms per step unless noted):

| Commit | Change | Batch 1 | Batch 4 | Batch 16 |
| --- | --- | --: | --: | --: |
| `87f99ff` | Fused q/k norm and RoPE at every row count (one op of 28-47 us for the 68-120 us ttnn chain), decode inputs built on one torch thread | - | 25.85 -> 24.65 | 56.05 -> 49.81 |
| `7c565e5` | Router top-k shared over the router cores (op 145 -> 69 us at 16 rows), outputs unchanged | - | 24.64 -> 24.28 | 49.80 -> 47.56 |
| `c375703` | Batched prefill of a scheduler step's requests | - | serving 128.3 -> 139.0 tok/s | serving 222.0 -> 280.2 tok/s, TTFT 2.92 -> 1.39 s |
| `0875726` | Decode SDPA reads 128 cache positions per chunk | 18.33 -> 18.03 at 2k positions | - | 55.76 -> 55.37 at 2k |
| `41ef565` | LM head as stream kernels placed per chip (op 2.41 ms on chip 1 -> 1.68 ms) | 16.95 -> 15.84 | 24.10 -> 23.23 | 49.04 -> 47.61 |
| `ed12310` | Expert streams placed per chip | 15.88 -> 15.75 | 23.21 -> 22.78 | 47.76 -> 47.34 |
| `1b4b150` | o_proj through the down stream into compact vectors, pack op removed | 15.79 -> 15.70 | 22.77 -> 22.50 | 47.48 -> 46.49 |

Batched prefill alone: fifteen 144-token prompts 3.30 -> 1.45 s, four 512-token prompts 1.85 -> 1.32 s, two 1,500-token prompts 1.99 -> 1.88 s. The serving gains of `c375703` include `7c565e5`, which landed between the two runs.

Profiling findings:

- A 16-row step's blocking trace replay was 30.2 ms after these changes; per layer the experts take 450-560 us (at the DRAM limit for the step's distinct experts), the tails 110-130 us, the two all-gathers 85-180 us, QKV 70 us, SDPA and cache updates 78 us.
- The chips are harvested differently. ttnn's DRAM-sharded matmuls place their readers from chip 0's DRAM map, so chip 1 ran the LM head 0.55 ms slower per step, and the all-gathers made both chips wait for it. The per-chip stream LM head removed that gap.
- At 16 users, py-spy put 89% of `decode_forward` in `LmHead.read`: the wait for the device plus the 16 MB logit readback (4.3 ms; chip 1's half comes over Ethernet). Python input building took 5.8% and sampling 3.4%; vLLM's own step was about 2% of the engine loop.
- A 32-token prefill is device-bound (36 ms wall, 33 ms device for 6 layers): the expert-major `sparse_matmul` reads at about 100 GB/s and its row gathers are mostly padding, so short prefills pay about 110 ms of fixed cost.
- The tails take 30-40 us whether they run on 3, 6 or 11 cores per row: they are latency-bound.

Accuracy (decode KL against fp32, mean of 3 seeds): batch 1 0.4436 -> 0.4392 over the round (the SDPA chunk change moved it; the LM head and per-chip changes left it unchanged, the o_proj stream moved single seeds by up to 0.014 with the same mean). Batch 4 0.4338 -> 0.4410 and batch 16 0.4317 -> 0.4268 after the fused RoPE; the router change left batch 4 bit-identical. Batched against one-by-one prefill, the last prompt row's KL over 20 rows per seed was 0.274 against 0.292 (6 seeds, signs split 3 and 3); the batched path differs from one by one only from the o_proj matmul on, whose rounding depends on M. `serve_check.py` passed every check on `1b4b150`, including sixteen concurrent requests that match their single runs token for token.

Tried and not kept:

| Attempt | Result |
| --- | --- |
| BFP4 LM head (`Precision(lm_head=bfp4)`) | Batch 1 17.05 -> 15.95 ms before the stream LM head, but KL 0.4436 -> 0.4572, worse in every paired seed; left as an option |
| Reading both chips' logits from two threads | 4.63 against 4.21 ms at 16 rows: the reads serialize |
| `ttnn.argmax` / `ttnn.max` on the logits | 1.5-2.4 ms (75 ms for a tiled argmax), no cheaper than the readback |
| SDPA decode grids (8, 2) and (8, 4) | Within 0.1-0.3 ms of (8, 8) |
| Stream LM head with 3 readers or 4-tile groups | 1.84-1.90 ms against 1.68 with 2 readers and 3-tile groups |
| `paged_fused_update_cache` for K and V | Has no paged-geometry override, which the shared K/V buffers need |
| QKV through the down stream | Its 128 or 160 output tiles would need padding to 144 or 168 to split over 12 banks, up to 12.5% more bytes |

**Decision.** All seven changes kept; `serve-gemma4` serves 16 users. Review by Sol 6.2 (medium) found a write past the q/k RoPE scratch buffer and a torch thread-count race (`b989662`), the Q4_0 prefill routing (`53cd396`), and an SDPA chunk that could read past an odd-width page table, a doubled host copy of the LM head weight and an unused dispatch branch (`4750c35`); all fixed.

**Data.** `2026-11-23-decode-serving-round-3/`: `decode/` (the final standalone runs), `accuracy/` (per-change runs, named after the change), `serve/<label>/` (bench results and `summary.txt`). The earlier per-change runs were logs in `artifacts/`.
