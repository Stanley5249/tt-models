# Pi coding agent against the server

**Question.** Does a coding agent (Pi) work end to end against `serve-gemma4`: tool calls, reasoning output, multi-turn requests?

**Setup.** `a527dad` with the gemma4 tool and reasoning parsers in `pixi.toml`. `pixi run serve-gemma4`, then `pi --offline --no-session --model tt-vllm/gemma4 --tools read,edit,bash --mode json -p <task>`: read a random value from a scratch file, edit another file, run a check script with Bash.

**Result.** The tool sequence read, read, edit, bash succeeded; the final answer matched the check script, and 5 non-empty thinking blocks arrived. All 5 chat requests returned HTTP 200. The first server start hung in ARC firmware init on device 0 and needed a card reset (cause unknown). Warnings seen: a HiFi4 with FP32 accumulation precision warning at warmup, one possible trace buffer overwrite at warmup (later found and fixed, see [trace safety](2026-11-21-trace-safety.md)), semaphores falling back to L1. The adapter at that time refused chunked prefill and prefix caching.

**Decision.** Tool calling and reasoning work through vLLM's parsers, so they stay in `serve-gemma4`. The adapter limits listed fed the [serving investigation](2026-11-21-serving-investigation.md).

**Data.** `2026-11-19-pi-agent-validation/report.zh-TW.md` (the original report, in Traditional Chinese). Its server logs were local and are not in the repository.
