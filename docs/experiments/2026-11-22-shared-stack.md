# Shared stack from tt-scope v0.5.0

**Question.** Can the runtime stack come from tt-scope's release instead of this manifest, and does the move from ttnn 0.81.1+conda.8 to 0.82.0+conda.1 that comes with it keep numerics and speed?

**Setup.** Baseline `0d46091` (main: ttnn 0.81.1+conda.8, tt-scope v0.4.0, vllm-tt-plugin `0c12fd0`), shared stack on `build/shared-stack` (tt-scope v0.5.0 with `[stable,vllm]`: ttnn 0.82.0+conda.1, the same torch 2.11.0+cpu, vLLM 0.32.0 and vllm-tt-plugin `0c12fd0`), same day, CPU governor `performance`. Commands from the repository root, in this order on both sides:

    pixi run pytest -m device
    pixi run python scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json
    pixi run python scripts/accuracy/compare.py --decode-steps 32 --offsets 0 2048 --trace --batch <1|4>
    pixi run python scripts/bench/decode.py --batch <1|4> --steps 128 --trace     # 3 runs each
    pixi run python scripts/bench/prefill.py
    just serve-gemma4, then scripts/serve/serve_check.py and bench_tis.sh c1-128-128-n8 c4-128-128-n16

**Result.** The device tests pass and greedy outputs match 4/4 on both. `compare.py` is identical at batch 1 and 4, every one of the 66 rows (mean KL 0.2637 and 0.2654). `serve_check.py` passes every check on both. Standalone decode, median of 128 steps, mean of 3 runs; prefill, min of 2:

| Measure          |           main |   shared stack |
| ---------------- | -------------: | -------------: |
| Decode, batch 1  |       17.61 ms |       16.91 ms |
| Decode, batch 4  |       29.63 ms |       28.74 ms |
| Prefill, 128     |        0.199 s |        0.198 s |
| Prefill, 2048    |        1.283 s |        1.277 s |
| Serving TPOT, c1 | 18.10-18.16 ms | 17.46-17.47 ms |
| Serving TPOT, c4 | 30.66-30.78 ms | 29.61-29.96 ms |

`just serve-qwen3-8b` starts on the shared stack and answers a greedy request ("Paris"), so tt_transformers still loads after 0.82.0 moved it to `models.ttt_compat`. The decode runs varied by at most 0.25 ms within each side. A tt-scope capture of `scripts/profile/workload.py --batch 4 --decode-steps 2 --trace` now needs `--programs 20000`: at 8,000 and 12,000 the cores at (18,18) and (19,18) of both chips dropped markers. tt-scope v0.5.0 builds the ops report only on `tt-scope report`, into `report/ops_perf_results.csv`, which `trace_breakdown.py` now reads.

**Decision.** Kept. The manifest keeps only what pixi cannot take from a dependency: Python and the `tt-metal` conda package, activation, and the overrides.

**Data.** None; the runs are reproducible from the commands above.
