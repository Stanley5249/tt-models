# Single-user decode, round 2: custom streams and compact boundaries

**Question.** Can batch-1 decode get close to the DRAM roofline with custom kernels where ttnn ops stall: the expert sparse matmuls, the boundaries between layers, the router and the all-reduce?

**Setup.** Baseline `5887694`, final `f158df1`, CPU governor `performance` for both, measured back to back. tt-scope v0.4.0 `tracy` captures of `scripts/profile/workload.py --trace`, folded by `scripts/profile/trace_breakdown.py`. `trace_time.py` for blocking trace time, `final_standalone.sh` (last at `da811f2`) for 3-run standalone timing. CCL, LM head and router microbenchmarks `bench_ccl*.py`, `bench_lm_head.py`, `bench_router.py`, `bench_sparse_ksplit.py` (all last at `da811f2`).

**Result.** Trace execution 31.5 -> 15.6 ms, ops per step 2,199 -> 574, standalone decode 33.3 -> 17.5 ms, serving TPOT 35.2 -> 18.7 ms at one user. Decode KL 0.387 -> 0.368 over 6 seeds. Steps, each measured at its commit:

| Change                                                                       |      Trace exec |
| ---------------------------------------------------------------------------- | --------------: |
| LM head as 16 DRAM-sharded splits, softcap on host                           | 31.5 -> 30.9 ms |
| All-reduce as `all_gather_async` plus a local sum                            |         -> 30.1 |
| Expert streams: per-DRAM-bank reads of the routed experts                    |         -> 24.1 |
| Compact boundaries: FP32 residual as 3 dense tiles, one tail op per boundary |         -> 19.7 |
| Dense MLP through the stream kernels                                         |        -> 18.65 |
| Fused q/k RMSNorm and RoPE                                                   |         -> 17.7 |
| Router op with top-8 on the reducer RISC (71 -> 36 us)                       |         -> 16.7 |
| Router merged with MLP gate\|up on disjoint cores, FPU RMS sums              |         -> 15.6 |

Not kept: `all_reduce_async` and `num_links=2` (both hung the card), router as linear plus topk (161 us), K-split sparse matmul, merging two DRAM-bound streams into one program, DRAM-sharded QKV fed by the tail, `paged_fused_update_cache` (global layers only). The loop stopped after three attempts under 3%.

**Decision.** Kept. Batch 4 still ran the old path, which [batched expert streams](2026-11-21-batched-expert-streams.md) and [compact boundaries for 2-4 rows](2026-11-21-compact-boundaries-multi-row.md) later fixed. Remaining gap to the roofline: weight ops at 170-210 GB/s, about 5 ms of non-weight ops, host work, and chip 1 running 0.5 ms behind.

**Data.** `2026-11-19-decode-round-2/`: `report.md` (the original write-up with the per-layer table and roofline), trace breakdowns in `final/`, serving runs, and accuracy runs per step (`decode4x128-<step>-p*.json`).
