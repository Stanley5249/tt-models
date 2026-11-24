# Serving Gemma 4 26B-A4B on an N300

How the vLLM backend works, what it supports, and how to check a change. Measurements and the reasons behind each design live in [experiments](experiments/README.md); this guide links to them instead of repeating their numbers.

    just serve-gemma4                                # 127.0.0.1:8000, model "gemma4"
    pixi run python scripts/serve/serve_check.py     # request lifecycle checks

## Architecture

The adapter is `src/tt_models/models/gemma4/vllm_adapter.py`. `src/tt_models/vllm_plugin.py`, a `vllm.general_plugins` entry point, registers it and the tool parser in every vLLM process. The model code (`model.py`, `layer.py`, `attention.py`, `moe.py`) runs on a 1x2 mesh with tensor parallelism over the two chips.

### KV cache

vLLM owns every layer's KV cache ([hybrid KV cache](experiments/2026-11-21-hybrid-kv-cache.md)). Sliding layers are `SlidingWindowSpec` and global layers `FullAttentionSpec`; vLLM's hybrid manager forms 5 sliding groups and 1 global group with their own page tables, and the i-th layer of every group shares one K and one V buffer (`kv_cache.py`). The model keeps no per-request state, so preemption, resumption and batch-row moves need no adapter hooks. `GEMMA4_KV_POOL_GIB` sets the pool size per chip.

### Prefill

`Gemma4TT.prefill(tokens, start, tables)` computes positions `[start, len(tokens))` and writes only their K/V. `start` must lie on the block grid, the lcm of the group block sizes, so prefix hits and chunk ends never rewrite shared blocks; the adapter refuses a chunk alignment off the grid at startup. Long prompts run in pieces of at most `PREFILL_CHUNK` tokens, and every piece pads to a bucket (`config.prefill_rows`) whose programs are built at warmup ([length buckets](experiments/2026-11-21-prefill-length-buckets.md)). Experts run expert-major in one-tile chunks ([one-tile chunks](experiments/2026-11-21-one-tile-expert-chunks.md)). The requests of one scheduler step whose pieces fit `PREFILL_CHUNK` rows run as one pass (`Gemma4TT.prefill_batch`): projections, MLP and experts take every piece's rows, and each piece writes its K/V and attends over its own sequence ([round 3](experiments/2026-11-23-decode-serving-round-3.md)).

Prefix caching is vLLM's automatic prefix caching. Chunked prefill uses the plugin's scheduler with one decode step after every prefill chunk ([decode interleave](experiments/2026-11-21-decode-interleave.md)).

### Decode

Decode captures one metal trace per active row count (1 to `STREAM_ROWS`, 16). Every row count runs the same fused path: compact FP32 boundaries with one tail op per boundary (`boundary.py`), the router op (`router.py`), and the expert and MLP streams (`expert_stream.py`), where one op pair covers every expert slot of every row. The router lists the step's distinct experts, so an expert that several rows picked is read once ([shared expert reads](experiments/2026-11-23-shared-expert-reads.md)), and its cores share the rows' top-k. The fused q/k norm and RoPE op (`qk_rope.py`) covers every row count. o_proj and the dense MLP down run through the down stream (`DownStream`) straight into compact vectors, and the LM head streams its weight with its own kernels (`lm_head.py`). The streams and the LM head place their cores per chip, because the two chips are harvested differently. See [round 2](experiments/2026-11-19-decode-round-2.md), [batched streams](experiments/2026-11-21-batched-expert-streams.md), [compact boundaries](experiments/2026-11-21-compact-boundaries-multi-row.md) and [round 3](experiments/2026-11-23-decode-serving-round-3.md).

Batch 1 and larger batches still run some different kernels, which round differently, so a greedy request can continue differently at a near-tie when it shares decode steps with others. Both paths are equally accurate against fp32 ([batch paths](experiments/2026-11-21-batch-path-accuracy.md)).

Everything a trace touches is allocated before the first capture, and prefill reduces across chips with semaphores allocated at load ([trace safety](experiments/2026-11-21-trace-safety.md)). New code must keep both rules.

### Sampling

When every active decode row is greedy and needs no logprobs, each chip reduces its FP32 logits to local maxima. The host merges those candidates in vocabulary order, so an exact tie selects the lowest token ID. Greedy readback uses the same trace as full-logit decode. See [decode readback](experiments/2026-11-25-gemma4-decode-readback.md) and its [audit](experiments/2026-11-25-decode-readback-audit.md).

The host model-side sampler (`sample` in `vllm_adapter.py`) reads full logits for mixed or sampled batches. Sampling takes the top k <= 1,024 candidates and applies temperature, top-p and a seed keyed by (seed, position). Only the candidates get the final softcap ([lazy softcap](experiments/2026-11-21-lazy-softcap.md)). Logprobs and vLLM's host sampler get the whole capped row.

### Tool choice

`src/tt_models/models/gemma4/tool_parser.py` registers xgrammar's `gemma_4` structural tag, so `tool_choice="required"` and named tools are enforced ([tool choice](experiments/2026-11-21-tool-choice.md)). It needs the xgrammar override in `pyproject.toml`.

### Precision

Experts are BFP4 at LoFi; attention, MLP, LM head and KV cache are BFP8 (`Precision(lm_head=bfp4)` trades decode KL for speed, [round 3](experiments/2026-11-23-decode-serving-round-3.md)); the residual stream between layers is FP32 (`config.Precision`). The checkpoint is Q4_0, which BFP4 cannot hold ([weight formats](experiments/2026-11-17-weight-precision.md), [BFP4](experiments/2026-11-21-bfp4-mixed-precision.md)). `Precision(expert_q4=True)` reads the exact Q4_0 experts in batch-1 decode; it is for accuracy runs only, because prefill then runs one token at a time ([Q4_0 decode experts](experiments/2026-11-21-q4-decode-experts.md)).

## Capabilities

| Capability | Status |
| --- | --- |
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
| Greedy decode readback | Exact FP32 device argmax with host merge |
| Async decode, stochastic device sampling | Open |
| Q4_0 experts | Accuracy runs only |
| vLLM streaming-input sessions | No: they resume prefill off the block grid |
| Multimodal, speculative decoding, data parallel | No (no MTP heads in this checkpoint) |

## Checking a change

Run what the change can affect, from the repository root. `py` stands for `pixi run python`.

| What | Command |
| --- | --- |
| Format, lint, types | `just check` |
| Host logic | `just test` |
| Device paths, random weights | `just test -m device` |
| One kernel against torch or ttnn | `py scripts/kernels/<name>_check.py` |
| Greedy outputs | `py scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json` |
| Decode accuracy against fp32 | `py scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <k>` for k = 1..3 |
| Standalone speed | `py scripts/bench/decode.py --batch <B> --steps 128 --trace`, `py scripts/bench/prefill.py` |
| Serving lifecycle | `py scripts/serve/serve_check.py` against a running server |
| Preemption | `GEMMA4_KV_POOL_GIB=0.2 just serve-gemma4 --max-model-len 4096`, then `py scripts/serve/serve_check.py preempt` |
| Serving speed | `pixi run python scripts/serve/bench_tis.py <label> c1-128-128-n8 c4-128-128-n16` |

All kernel checks, one process each:

    for f in scripts/kernels/*_check.py; do pixi run python "$f" || break; done

Accuracy needs 3 seeds and speed a same-day baseline ([experiments](experiments/README.md)). A numerics-preserving change should leave `compare.py`'s metrics identical.

## Profiling

    pixi run tt-scope tracy scripts/profile/workload.py --programs 20000 -- --batch 4 --decode-steps 2 --trace
    pixi run tt-scope report "artifacts/runs/$RUN"                                              # RUN: the capture's folder name
    pixi run python scripts/profile/trace_breakdown.py "artifacts/runs/$RUN" --phase decode1

`trace_breakdown.py` recomputes op time and gaps from device firmware cycles, because tt-scope's op-to-op latency is host time under trace replay ([tt-scope findings](experiments/2026-11-19-tt-scope-findings.md)). Other tools: `scripts/bench/trace_time.py` (blocking trace time), `scripts/profile/memory_report.py` (per-chip DRAM), `scripts/bench/soak.py` (DRAM growth over many traced steps), and server logs with `GEMMA4_TIMING=1` (per-chunk prefill and per-step decode time).

## Open work

- **Q4_0 serving.** Needs an expert-major prefill matmul that dequantizes inside, batch 2-4 down kernels, compact step tiles, and DRAM for the steps ([prefill dequantization](experiments/2026-11-21-q4-prefill-dequant.md)).
- **Stochastic device sampling and async decode.** A device top-k could reduce full-logit readback for sampled batches. Async decode also needs the embedding on the device. See [device sampling](experiments/2026-11-21-device-sampling.md).
- **Multi-row decode.** 9-15 rows concatenate heads through a permute. More than 16 rows needs the stream kernels to use a tile's bottom faces. The tails (30-40 us each) are latency-bound whatever their core count.
- **Host overhead.** Input build, readback and sampling run outside the device each step; only async decode removes most of it.
- **Prefill.** The all-gathers are bound by one Ethernet link, and the gate|up sparse matmul uses a fraction of the cores.
- **Tool choice upstream.** Drop `tool_parser.py` once vLLM sets the structural tag for gemma4 (vllm-project/vllm#53363).
- **Quality.** More GPQA questions or seeds before comparing precision choices ([GPQA baseline](experiments/2026-11-21-gpqa-baseline.md)).
