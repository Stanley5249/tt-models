# Compact boundaries for 2-4 decode rows

**Question.** After the batched streams, what else makes a batch-4 step slow, and does batch 1's compact boundary path generalize to 2-4 rows?

**Setup.** `a712ab9`, merged in `91d4461`. tt-scope captures of one traced batch-1 and batch-4 step on `main` and after. Tail cost by `scripts/kernels/boundary_check.py --rows 1 4 --tail-cores 6` (trace replay); standalone steps by `scripts/bench/decode.py --trace` (128 steps); serving by `scripts/serve/bench_tis.sh` against both servers, same day; accuracy 3 seeds per batch size.

**Result.** The batch-4 capture on `main` showed 32.4 ms of kernel time and 1,778 ops on chip 0. About 7 ms went to glue between matmuls (211 LayerNorms, 300 binary ops, 423 layout conversions, slices, concats) and 3.3 ms to 60 all-gathers moving 32-row tiles for 4 real rows. The compact path keeps one FP32 vector per row (3 dense tiles), the o_proj pack and the down streams write each row's vector, and one tail op per boundary does the chip sum, every RMSNorm, the residual adds and the layer scalar.

| Tail variant, 4 rows                         | Attention tail | FF tail |
| -------------------------------------------- | -------------: | ------: |
| One row on its own                           |          27 us |   37 us |
| All rows on every core                       |          69 us |  111 us |
| One core group per row                       |          53 us |  105 us |
| One core group per row, reading only its row |          34 us |   48 us |

Kernel time per batch-4 step 32.4 -> 24.8 ms, ops 1,778 -> 784. Standalone steps: batch 2 28.85 -> 22.67 ms, batch 3 31.80 -> 26.44, batch 4 34.95 -> 29.49, batch 1 unchanged. Serving, 128 in / 128 out:

| Users |  TPOT main -> new | Output tok/s main -> new |
| ----- | ----------------: | -----------------------: |
| 1     | 18.07 -> 18.08 ms |             50.6 -> 50.6 |
| 2     |   29.6 -> 23.1 ms |             62.8 -> 79.1 |
| 4     | 36.17 -> 30.77 ms |            96.9 -> 111.3 |

Accuracy: batch 1 bit-identical (KL 0.457); batch 4 KL 0.449 against 0.423 for the old path, inside the noise.

**Decision.** Kept; the row-form decode path was removed. The batch-4 step now leads with the experts (8.3 ms, at the DRAM limit for 32 slots), the merged router and MLP op (2.1 ms), the LM head (1.9 ms) and QKV (1.7 ms).

**Data.** `2026-11-21-compact-boundaries-multi-row/`: per-op tables (`decode-b1-now.txt`, `decode-b4-main.txt`, `decode-b4-compact.txt`), `compact-rows/` (accuracy), `perf-profiling/` and `perf-profiling-base/` (serving). The 2-user `main` serving files were lost to a mislabeled directory; those numbers come from a run log.
