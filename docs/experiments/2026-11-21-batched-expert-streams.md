# Expert and MLP streams for 1-4 decode rows

**Question.** Why is a batch-4 decode step 51 ms, and can the batch-1 stream kernels serve 2-4 rows?

**Setup.** `0e65f24`. tt-scope `tracy` capture of one traced batch-4 step before and after:

    pixi run tt-scope tracy scripts/profile/workload.py --programs 8000 -- --batch 4 --decode-steps 2 --trace
    pixi run python scripts/profile/trace_breakdown.py artifacts/runs/<run> --phase decode1

Correctness by `scripts/kernels/expert_stream_check.py --rows 4` against the prefill expert path.

**Result.** Before: 2,348 ops and 50.9 ms of kernel time per chip, with the indexed `sparse_matmul` at 20.5 ms (about 85 GB/s for 32 expert slots), gate and up slicing 3.3 ms, and only 0.86 ms of gaps, so kernel time had to fall. The stream kernels already multiply whole 32-row tiles, so they were extended: slot `s = 8 t + j` is token t's j-th expert, the gate|up writer puts the routing weight in that token's row, and the down kernel's slot sum leaves each token's experts in its own row. After: 1,898 ops, 31.5 ms; expert gate|up 191 us and down 86 us per layer (about 204 and 226 GB/s). Standalone batch 4: 56.49 -> 35.42 ms. Per-row PCC against the prefill path 0.99972-0.99986.

**Decision.** Kept; the dense MLP uses the same kernels with every slot serving every row. The router, norms and LM head then led the batch-4 step, which [compact boundaries](2026-11-21-compact-boundaries-multi-row.md) and the [router op](2026-11-21-router-op-multi-row.md) addressed.

**Data.** `2026-11-21-batched-expert-streams/`: per-op tables `decode-b4-before.txt` (the KV rewrite just before this change) and `decode-b4-after.txt`.
