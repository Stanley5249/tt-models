Completed the Gemma 4 serving investigation. The report covers the capability audit, upstream compatibility, hardware measurements, and a prioritized implementation plan. Implementation has not started.

- **Prefill blocks active streams.** An 8,192-token arrival caused a **5.66-second gap** in an existing stream while the adapter spent **5.61 seconds** in synchronous prefill. The TT scheduler already alternates prefill and decode steps. Supporting continuation would let it alternate at chunk boundaries.
- **Multi-user decode needs its own optimization.** The new two-user run spent about **40–41 ms of each 42 ms decode step inside the model call**. Most of the steady-state penalty is in the wider decode path. The plan preserves the optimized single-row path and profiles widths two and four before choosing a fix.
- **Repeated context dominates the synthetic agentic turns.** Both coding chat turns recomputed roughly 8K prompt tokens and spent **14.7 seconds in prefill**. The earlier Pi run verified tool calling. Pi’s total latency breakdown remains unmeasured.
- **Greedy repeatability needs investigation first.** Identical isolated raw-token requests produced different outputs, while the short text control matched. The evidence establishes a repeatability failure for that input. It does not establish slot corruption.

The plan starts with correctness and lifecycle checks, then prioritizes chunked prefill and multi-row decode profiling. It evaluates prefix caching against measured N300d memory headroom before changing KV ownership.

[Raw measurements](measurements.json) and [prefill timings](prefill-timing.log) are saved alongside the report. The server stopped and released the device. No reset or dependency update was needed. The three investigation files and the pre-existing workspace changes are uncommitted.

Implementation, results and remaining work: [docs/serving.md](../../serving.md).
