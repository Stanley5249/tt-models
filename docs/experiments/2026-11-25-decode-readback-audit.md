# Audit of the greedy readback, shared RoPE tiles and one-worker gathers

**Question.** Do the changes of the [decode reassessment](2026-11-25-gemma4-decode-readback.md) hold up: are outputs unchanged, is serving safe, and is each gain worth its code?

**Setup.** 2026-11-25, CPU governor `performance`, N300d. The four commits ran from `git archive` copies through `PYTHONPATH`, in the order `f87908f`, `bf41a37`, `9d2b250`, `e87dbd9` and back, at 128-token distinct prompts, three seeds, 63 steps each, one torch thread as in vLLM. The serving runs compare `f87908f` with the audited tree through one `just serve-gemma4` each, with one discarded warm-up and three seeds per point. Server order was audited tree first, then baseline.

**Result.** Mean over both passes of the per-seed median full-logit step, in ms. The argmax is not included, so `bf41a37` is a control here.

| Rows | `f87908f` | `bf41a37` | `9d2b250` (RoPE tiles) | `e87dbd9` (gathers) |
| ---: | --------: | --------: | ---------------------: | ------------------: |
|    1 |     15.43 |     15.40 |                  15.39 |               15.21 |
|    2 |     17.34 |     17.34 |                  17.30 |               17.19 |
|    4 |     20.66 |     20.68 |                  20.72 |               20.63 |
|   16 |     37.64 |     37.68 |                  37.18 |               37.12 |

Every step's full FP32 logits hashed to the same digest in all four commits and both passes, for all 12 row-count and seed pairs.

Greedy step on `e87dbd9`, ms, with the three token readouts interleaved at every step. Every row kept its own continuation.

| Rows | Full logits, torch argmax | Full logits, NumPy argmax | Device maxima |
| ---: | ------------------------: | ------------------------: | ------------: |
|    1 |                     15.37 |                     15.25 |         15.24 |
|    4 |                     22.18 |                     21.69 |         21.20 |
|   11 |                     35.31 |                     33.98 |         32.20 |
|   16 |                     41.24 |                     39.54 |         36.01 |

The device maxima picked the same token as the full-logit argmax in all 6,048 row-steps. A host NumPy argmax would match them at one row, but leaves 3.5 ms at 16 rows.

Serving, mean of three seeds (tok/s, TPOT):

| Point             | `f87908f`       | Audited tree    |
| ----------------- | --------------- | --------------- |
| 1 user, 128/128   | 54.95, 16.49 ms | 56.16, 16.09 ms |
| 4 users, 128/128  | 144.05, 23.67   | 151.35, 22.41   |
| 8 users, 128/128  | 212.13, 31.23   | 226.17, 28.92   |
| 16 users, 128/128 | 288.77, 44.32   | 316.76, 39.36   |

The earlier record's eight-user baseline of 190.94 tok/s includes one run that compiled programs cold. Without such a run the gain is 212 to 226 tok/s (6.6%), not 18%.

`serve_check.py` passed. Sixteen prompts, as chat and as raw text, ran greedy, seeded sampled, with logprobs and with a repetition penalty: all 128 responses sent alone matched `f87908f` token for token and repeated exactly. Sixteen concurrent chat requests matched `f87908f` in 78 of 80 responses over all-greedy and mixed batches. Concurrent raw text differed between servers, but not because of these changes: 16 repeats on each server gave the same two main outcomes (9 and 3 of 16 on the audited tree, 8 and 4 on `f87908f`). Concurrent requests do depart from their solo runs on both versions, because batched prefill and different row counts round differently. `serve_check.py`'s four prompts did not depart.

Findings in the code:

- The readback op was built on first use. vLLM's untraced warmup built it before any capture, so the server was safe, but any other caller could put its kernels in trace memory, against the [trace safety](2026-11-21-trace-safety.md) rule. `_prepare_traces` now builds it for every row count, which also removes the special case from the adapter's warmup. The device test that covers this fails on `e87dbd9`.
- The result buffer is still allocated per step, after captures. It can only overlap trace intermediates, and the queue orders its write and blocking read before the next replay.
- `sample()`'s all-greedy branch became unreachable and is removed.

**Decision.** All three changes kept. The readback is worth its kernel from four rows on. The RoPE change is 0.4 to 0.5 ms at 16 rows only, about 1%, with less code in `model.py` than before. The gather change is one expression worth 0.19 ms at one row and 0.11 at two. Sampled, mixed and logprob batches gain nothing from the readback: one such request makes all rows read full logits, and the checkpoint's default generation settings sample.

**Not checked.** FP32 accuracy was not rerun, because the logits are bit-identical. Speed beyond about 300 positions and preemption were not rerun.

**Data.** `2026-11-25-decode-readback-audit/`: `standalone.jsonl`, `serving.json`, and the two one-off scripts as `.txt`.
