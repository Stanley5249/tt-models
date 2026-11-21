# Repository restructure to tt-models

**Question.** Can the code move into an installable `tt_models` package, with
large modules split, scripts regrouped and a vLLM plugin entry point, without
changing numerics or speed?

**Setup.** Baseline `348b9fe` (main), restructure `46074a1..271f3f0`, same
day. Commands from the repository root:

    pixi run python scripts/accuracy/compare.py --decode-steps 32 --offsets 0 2048 --trace --batch <1|4>
    pixi run python scripts/bench/decode.py --batch <1|4> --steps 128 --trace     # 3 runs each
    pixi run python scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json
    pixi run pytest -m device
    pixi run serve-gemma4, then scripts/serve/serve_check.py and bench_tis.sh c1-128-128-n8 c4-128-128-n16

The baseline ran `scripts/bench_decode.py` and `compare.py` at their old paths.

**Result.** `compare.py` metrics are identical to the baseline at batch 1 and
4 (every KL, top-1 and PCC value; only paths and timings differ). Greedy
outputs match 4/4. Standalone decode, median of 128 steps, mean of 3 runs:

| Batch | main | restructure |
|---:|---:|---:|
| 1 | 17.83 ms | 17.69 ms |
| 4 | 29.81 ms | 29.70 ms |

The device tests pass, and `serve_check.py` passes every check, including the
prefix-cache check that failed on `main`: its prompt was built from
repository files, and it now uses the checkpoint's model card. Serving, one
run of 3 repetitions: TPOT 17.96-18.00 ms at 1 user and 30.33-30.78 ms at 4,
in line with [compact boundaries](2026-11-21-compact-boundaries-multi-row.md).

**Decision.** Merged. ttnn stays at 0.81.1+conda.8: 0.82.0 ships as a thin
wheel plus a `tt-metal` conda package, moves tt_transformers to
`models.ttt_compat`, and has no matching tt-scope release yet. vLLM stays at
0.32.0, which vllm-tt-plugin pins.

**Data.** None; the runs are reproducible from the commands above.
