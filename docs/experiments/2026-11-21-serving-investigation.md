# Serving investigation

**Question.** What blocks good multi-user and agentic serving: capability gaps, latency, or correctness?

**Setup.** `a527dad`-era server on vLLM 0.31. Probes against `serve-gemma4`: two concurrent streams, an 8,192-token arrival during a stream, repeated identical raw-token requests, and an API smoke test.

**Result.**

- An 8K-token arrival stopped a running stream for 5.66 s while the adapter ran 5.61 s of unchunked prefill.
- Two users spent about 40-41 ms of each 42 ms decode step in the model: the wide decode path, not vLLM, was slow.
- Agentic chat turns recomputed about 8K prompt tokens each (14.7 s of prefill), because there was no prefix cache.
- Identical raw-token requests produced different outputs.

**Decision.** In order: find the repeatability bug ([trace safety](2026-11-21-trace-safety.md)), move every KV layer into vLLM for prefix caching and chunked prefill ([hybrid KV cache](2026-11-21-hybrid-kv-cache.md)), then profile and fix multi-row decode ([batched expert streams](2026-11-21-batched-expert-streams.md)).

**Data.** `2026-11-21-serving-investigation/`: `summary.md` (the original summary), `measurements.json`, `prefill-timing.log`.
