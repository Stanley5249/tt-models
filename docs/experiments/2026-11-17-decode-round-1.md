# Decode and prefill, round 1

**Question.** Where does a decode step of the first working model go, and which ttnn-level changes cut it without losing accuracy?

**Setup.** Baseline `5101e0e`, final `5887694`, one N300d, BFP4 experts and BFP8 elsewhere. `scripts/bench/decode.py --trace` for standalone decode, `bench.sh` (last at `7f72451`) for vLLM serving, `compare.py --perturb` with 3 seeds for accuracy. Microbenchmarks `bench_sparse_matmul.py`, `bench_dram_sharded.py`, `bench_norm.py` and `bench_moe_compute.py` (all last at `7f72451`).

**Result.** Standalone batch-1 decode 49.7 -> 32.9 ms, serving TPOT 74.8 -> 35.2 ms at one user, 2,048-token TTFT 5.36 -> 1.46 s. Decode KL 0.390 -> 0.384 and prefill KL 0.226 -> 0.228 (3 seeds each), so accuracy held. The largest single wins were row-major logits (readback 9.1 -> 0.6 ms), a fused router and wider gate|up grid, fused QKV post-processing, expert-major prefill (TTFT 3.6x) and decoding only the active rows. The full table and the per-layer bottlenecks are in `report.md`.

Not kept: a 4-way LM head split, DRAM-sharded QKV (slower with its conversion), `addcmul` in RoPE, `moe_compute` (needs expert parallelism, 132 us per layer at batch 1) and BFP8 experts (do not fit).

**Decision.** All listed changes kept. The remaining time sat in the expert sparse matmuls (about 80 GB/s), which led to the custom streams of [round 2](2026-11-19-decode-round-2.md).

**Data.** `2026-11-17-decode-round-1/`: `report.md` (the original write-up), serving runs (`baseline-*`, `opt1`, `final`), standalone decode JSON, and the accuracy runs of each commit.
