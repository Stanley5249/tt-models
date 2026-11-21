# Serving Gemma 4 26B-A4B on an N300

How the vLLM backend works, what it supports, and how to check a change.
Measurements and the reasons behind each design live in
[experiments](experiments/README.md); this guide links to them instead of
repeating their numbers.

    pixi run serve-gemma4                            # 127.0.0.1:8000, model "gemma4"
    pixi run python scripts/serve/serve_check.py     # request lifecycle checks

## Architecture

The adapter is `src/tt_models/models/gemma4/vllm_adapter.py`. vllm-tt-plugin
loads it through `TT_MODEL_CLASS_OVERRIDES` in `pixi.toml`, and the model
code (`model.py`, `layer.py`, `attention.py`, `moe.py`) runs on a 1x2 mesh
with tensor parallelism over the two chips.

### KV cache

vLLM owns every layer's KV cache
([hybrid KV cache](experiments/2026-11-21-hybrid-kv-cache.md)). Sliding
layers are `SlidingWindowSpec` and global layers `FullAttentionSpec`; vLLM's
hybrid manager forms 5 sliding groups and 1 global group with their own page
tables, and the i-th layer of every group shares one K and one V buffer
(`kv_cache.py`). The model keeps no per-request state, so preemption,
resumption and batch-row moves need no adapter hooks. `GEMMA4_KV_POOL_GIB`
sets the pool size per chip.

### Prefill

`Gemma4TT.prefill(tokens, start, tables)` computes positions
`[start, len(tokens))` and writes only their K/V. `start` must lie on the
block grid, the lcm of the group block sizes, so prefix hits and chunk ends
never rewrite shared blocks; the adapter refuses a chunk alignment off the
grid at startup. Long prompts run in pieces of at most `PREFILL_CHUNK`
tokens, and every piece pads to a bucket (`config.prefill_rows`) whose
programs are built at warmup
([length buckets](experiments/2026-11-21-prefill-length-buckets.md)). Experts
run expert-major in one-tile chunks
([one-tile chunks](experiments/2026-11-21-one-tile-expert-chunks.md)).

Prefix caching is vLLM's automatic prefix caching. Chunked prefill uses the
plugin's scheduler with one decode step after every prefill chunk
([decode interleave](experiments/2026-11-21-decode-interleave.md)).

### Decode

Decode captures one metal trace per active row count (1 to `STREAM_ROWS`).
Every row count runs the same fused path: compact FP32 boundaries with one
tail op per boundary (`boundary.py`), the router op (`router.py`), and the
expert and MLP streams (`expert_stream.py`), where one op pair covers every
expert slot of every row. Batch 1 keeps two specializations: the fused q/k
norm and RoPE op (`qk_rope.py`) and a one-token core placement in the
streams. See [round 2](experiments/2026-11-19-decode-round-2.md),
[batched streams](experiments/2026-11-21-batched-expert-streams.md) and
[compact boundaries](experiments/2026-11-21-compact-boundaries-multi-row.md).

Batch 1 and batches 2-4 still run some different kernels, which round
differently, so a greedy request can continue differently at a near-tie when
it shares decode steps with others. Both paths are equally accurate against
fp32 ([batch paths](experiments/2026-11-21-batch-path-accuracy.md)).

Everything a trace touches is allocated before the first capture, and
prefill reduces across chips with semaphores allocated at load
([trace safety](experiments/2026-11-21-trace-safety.md)). New code must keep
both rules.

### Sampling

The model-side sampler (`sample` in `vllm_adapter.py`) runs on the host after
the logits are read back: greedy is an argmax, sampling takes the top
k <= 1,024 candidates and applies temperature, top-p and a seed keyed by
(seed, position). Only the candidates get the final softcap
([lazy softcap](experiments/2026-11-21-lazy-softcap.md)); logprobs and vLLM's
host sampler still get the whole capped row. Device sampling and async
decode were evaluated and not built
([device sampling](experiments/2026-11-21-device-sampling.md)).

### Tool choice

`src/tt_models/models/gemma4/tool_parser.py` registers xgrammar's `gemma_4`
structural tag, so `tool_choice="required"` and named tools are enforced
([tool choice](experiments/2026-11-21-tool-choice.md)). It needs the
xgrammar override in `pixi.toml`.

### Precision

Experts are BFP4 at LoFi; attention, MLP, LM head and KV cache are BFP8; the
residual stream between layers is FP32 (`config.Precision`). The checkpoint
is Q4_0, which BFP4 cannot hold
([weight formats](experiments/2026-11-17-weight-precision.md),
[BFP4](experiments/2026-11-21-bfp4-mixed-precision.md)).
`Precision(expert_q4=True)` reads the exact Q4_0 experts in batch-1 decode;
it is for accuracy runs only, because prefill then runs one token at a time
([Q4_0 decode experts](experiments/2026-11-21-q4-decode-experts.md)).

## Capabilities

| Capability | Status |
|---|---|
| Continuous batching | Up to `STREAM_ROWS` concurrent requests; more rows are refused at load |
| Prefix caching | Supported, block-grid hit granularity |
| Chunked prefill with decode interleave | Supported |
| Preemption and resumption | Supported (`serve_check.py preempt`) |
| Streaming, stop sequences, EOS, cancellation | Supported |
| Temperature, top-k <= 1,024, top-p, seed | Model-side sampler on the host; a seeded request can draw differently when a co-batched request forces vLLM's host sampler |
| Penalties, min-p, larger top-k, logprobs | vLLM's host sampler (per-step fallback) |
| Structured output, tool calls, reasoning parser | Through vLLM (host grammar, gemma4 parsers) |
| `tool_choice="required"` or a named tool | Enforced by `tool_parser.py` |
| `n > 1`, `logit_bias`, `echo` | Supported |
| Embeddings, `prompt_logprobs` | No (generative model; the plugin refuses prompt logprobs) |
| Async decode, device sampling | No (evaluated, see above) |
| Q4_0 experts | Accuracy runs only |
| vLLM streaming-input sessions | No: they resume prefill off the block grid |
| Multimodal, speculative decoding, data parallel | No (no MTP heads in this checkpoint) |

## Checking a change

Run what the change can affect, from the repository root:

| What | Command |
|---|---|
| Host logic | `pixi run pytest` |
| Device paths, random weights | `pixi run pytest -m device` |
| Kernels against torch or ttnn | `scripts/kernels/*_check.py` |
| Greedy outputs | `scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json` |
| Decode accuracy against fp32 | `scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <k>` for k = 1..3 |
| Standalone speed | `scripts/bench/decode.py --batch <B> --steps 128 --trace`, `scripts/bench/prefill.py` |
| Serving lifecycle | `scripts/serve/serve_check.py` against a running server |
| Preemption | `GEMMA4_KV_POOL_GIB=0.2 pixi run serve-gemma4 --max-model-len 4096`, then `serve_check.py preempt` |
| Serving speed | `scripts/serve/bench_tis.sh <label> c1-128-128-n8 c4-128-128-n16` |

Prefix every command with `pixi run python` (or `pixi run bash` for `.sh`).
Accuracy needs 3 seeds and speed a same-day baseline
([experiments](experiments/README.md)). A numerics-preserving change should
leave `compare.py`'s metrics identical.

## Profiling

    pixi run tt-scope tracy scripts/profile/workload.py --programs 8000 -- --batch 4 --decode-steps 2 --trace
    pixi run python scripts/profile/trace_breakdown.py artifacts/runs/<run> --phase decode1

`trace_breakdown.py` recomputes op time and gaps from device firmware cycles,
because tt-scope's op-to-op latency is host time under trace replay
([tt-scope findings](experiments/2026-11-19-tt-scope-findings.md)). Other
tools: `scripts/bench/trace_time.py` (blocking trace time),
`scripts/profile/memory_report.py` (per-chip DRAM),
`scripts/bench/soak.py` (DRAM growth over many traced steps), and server logs
with `GEMMA4_TIMING=1` (per-chunk prefill and per-step decode time).

## Open work

- **Q4_0 serving.** Needs an expert-major prefill matmul that dequantizes
  inside, batch 2-4 down kernels, compact step tiles, and DRAM for the steps
  ([prefill dequantization](experiments/2026-11-21-q4-prefill-dequant.md)).
- **Device sampling and async decode.** Needs a fused LM-head top-k and a
  device embedding.
- **Multi-row decode.** Expert slots shared by several users are read once
  per user; RoPE at 2-4 rows uses generic ops.
- **Host overhead.** Input build, readback, sampling and vLLM bookkeeping
  run outside the device each step; only async decode removes most of it.
- **Prefill.** The all-gathers are bound by one Ethernet link, and the
  gate|up sparse matmul uses a fraction of the cores.
- **Tool choice upstream.** Drop `tool_parser.py` once vLLM sets the
  structural tag for gemma4 (vllm-project/vllm#53363).
- **Quality.** More GPQA questions or seeds before comparing precision
  choices ([GPQA baseline](experiments/2026-11-21-gpqa-baseline.md)).
