# Serving Gemma 4 26B-A4B on N300 with vLLM

How the vLLM backend in `gemma4_tt/vllm_model.py` works, what it supports,
and how it was validated. Branch `feat/serving`, vLLM 0.32.0,
vllm-tt-plugin `0c12fd0`, ttnn 0.81.1+conda.8, one N300d (two Wormhole chips).
`docs/gemma4-serving-investigation.md` holds the investigation that started
this work.

    pixi run serve-gemma4                       # 127.0.0.1:8000, model "gemma4"
    pixi run python scripts/serve_check.py      # request lifecycle checks

## Architecture

**KV cache: vLLM's hybrid manager owns every layer.** The 25 sliding layers
are `SlidingWindowSpec` (window 1024) and the 5 global layers
`FullAttentionSpec`. vLLM groups them into 5 sliding groups and 1 global
group, each with its own page table. A block id belongs to one group at a
time, so the i-th layer of every group shares one K and one V buffer
(`gemma4_tt/tt/kv_cache.py`). To keep pages the same size, vLLM gives global
layers 128-token blocks (1 KV head of 512 per chip) next to 64-token sliding
blocks (4 heads of 256); each layer reads the shared buffer through its own
geometry. The model keeps no per-request state, so prefix caching,
preemption, resumption and row moves need no adapter hooks.

The pool is `KV_POOL_BYTES` = 2.5 GiB per chip: 3,860 blocks, 7.7 requests of
32K tokens. Weights take 8.0 of 11.87 GiB per chip. Blocks that running
requests do not use hold cached prefixes until vLLM evicts them.

**Prefill** (`Gemma4TT.prefill(tokens, start, tables)`) computes positions
`[start, len(tokens))` and writes their K/V. A chunk at position 0 attends
over its own K/V; a later chunk (chunked prefill, a prefix-cache hit) uses
chunked SDPA over the paged cache, with the chunk start in a device tensor so
new offsets reuse compiled programs. `start` must lie on the block grid (the
lcm of the group block sizes, 128 tokens), where vLLM puts prefix hits and the
plugin puts chunk ends; the adapter rejects a `prefill_chunk_alignment` off
that grid at startup. The model splits anything longer than 4,096 tokens into
pieces to bound activation memory.

**Decode** runs one traced step per active row count (1-4):

- 1 row: the fused path (compact FP32 boundaries, fused tails, router op,
  expert and MLP streams), unchanged by this work.
- 2-4 rows: generic attention and norms, but experts and the dense MLP run
  through the same DRAM-streaming kernels. One op pair covers all `8 x B`
  expert slots: slot `8 t + j` scales only token `t`'s row, so the down's slot
  sum leaves each token's experts in its own row.

**Trace safety.** A buffer allocated after a trace capture can overlap that
trace's intermediates and is overwritten when the trace replays.
`Gemma4TT._prepare_traces` allocates every batch size's inputs and runs every
batch size's decode once before the first capture, and prefill sums its
partials with `all_gather_async` on semaphores allocated at load time.
`ttnn.all_reduce` caches its own semaphores per new shape, and one cached
after a capture made the next prefill of that shape return garbage. Both bugs
predate this work; the second explains the investigation's identical requests
that disagreed from their first token.

## Capabilities

| Capability | Status |
|---|---|
| Continuous batching, dynamic admission, 1-4 concurrent requests | Supported (more than 4 rows is refused at load) |
| Prefix caching | Supported (vLLM automatic prefix caching, 128-token hit granularity) |
| Chunked prefill with decode interleave | Supported; one decode step after every 2,048-token chunk |
| Preemption and resumption | Supported (vLLM recomputes from cached blocks) |
| Streaming, stop sequences, EOS, cancellation | Supported |
| Sampling: temperature, top-k <= 1,024, top-p, seed | Model-side sampler on the host |
| Penalties, min-p, larger top-k, logprobs | vLLM's host sampler (the plugin falls back per step) |
| Structured output, tool calls, reasoning parser | Supported through vLLM (host grammar, gemma4 parsers) |
| `prompt_logprobs` | Not supported (refused by the TT plugin) |
| Async decode (`async_scheduling`) | Not supported: it needs tokens sampled on the device; `ttnn.topk` over a chip's 131,072 logits takes 38-71 ms |
| Device sampling, device penalties, device grammar | Not supported (same reason) |
| vLLM Python streaming-input sessions | Not supported: they can resume prefill off the block grid |
| Multimodal input, speculative decoding, data parallel | Not supported |

## Configuration

`serve-gemma4` in `pixi.toml` sets `--max-num-seqs 4`, `--max-model-len 32768`,
`--max-num-batched-tokens 2048` and `tt.decode_interleave_prefill_steps: 1`.
The interleave trade-off, measured with an 8,192-token prompt arriving during a
stream (steady state, 3 runs each):

| Chunk, prefill steps per decode step | Longest stream gap | 8K TTFT |
|---|---:|---:|
| 2,048, 2 (plugin default) | 3.38 s | 6.53 s |
| 2,048, 1 (configured) | 1.72 s | 6.56 s |
| 1,024, 1 | 1.00 s | 7.18 s |

Before chunked prefill the same arrival held the stream for 5.66 s.
`GEMMA4_TIMING=1` logs the model's share of every step.

## Validation

- `tests/smoke_chunked.py` (random weights): chunked prefill, a prefix hit on
  shared blocks, scheduler-shaped tables (freed sliding blocks and unallocated
  tail as the null block 0), that block 0 is never written, and that a start
  off the grid is refused. PCC >= 0.99998 against one full prefill.
- `tests/smoke_consistency.py` at batch 1-4 and past the window: decode vs
  prefill PCC 0.99996-0.99999. `tests/smoke_inactive.py`: 0.99998.
- `scripts/bench_expert_stream.py --rows 4`: the batched stream against the
  prefill expert path, PCC 0.99972-0.99986 per row.
- `scripts/greedy_check.py`: 4/4 prompts match the round-2 greedy tokens.
- `scripts/serve_check.py` against the server: concurrent requests, a raw-token
  prompt twice at temperature 0, an 8K arrival during a stream, a cancelled
  stream, a prefix-cache hit (confirmed through vLLM's metrics) against a
  salted miss, and seeded sampling. All pass with identical token ids.
- Teacher-forced decode against the fp32 reference, 3 seeds with matched
  noise: KL 0.421 (round 2) and 0.457 (now), top-1 0.777 and 0.758. The whole
  difference comes from the prefill reduction, the more exact of the two;
  prefill-only KL moved the other way. See
  `results/accuracy/matched-noise/README.md`.
- GPQA Diamond (`scripts/gpqa_eval.py`, upstream's QB2 settings: thinking off,
  temperature 1, seed 42): 26 of the first 40 questions (65%), 7 of the first
  10, none truncated (`results/quality/gpqa-feat-serving/`). Upstream gates its
  QB2 build, at BFP8 experts and BF16 dense weights, on 8 of the first 10. With
  40 sampled answers the 95% interval is about +-15 points, so this is a local
  baseline, not evidence of parity or of a gap.

## Performance

TIS-style client (`scripts/bench_tis.sh`, chat endpoint, random prompts,
ignore-EOS, greedy), mean of 3 runs. Before: `results/compare/tis-match/`
(round-2 code); after: `results/compare/feat-serving/`.

| Point | TTFT before | TTFT after | TPOT before | TPOT after | Output tok/s after |
|---|---:|---:|---:|---:|---:|
| 1 user, 128 in / 128 out | 237 ms | 238 ms | 18.17 ms | 18.45 ms | 49.5 |
| 1 user, 2,048 / 128 | 1.57 s | 1.48 s | 19.32 ms | 19.67 ms | 32.2 |
| 1 user, 8,192 / 128 | 6.95 s | 5.15 s | 19.69 ms | 20.02 ms | 16.6 |
| 2 users, 128 / 128 | - | 308 ms | ~42 ms (investigation) | 30.74 ms | 60.8 |
| 4 users, 128 / 128 | 736 ms | 694 ms | 57.95 ms | 37.75 ms | 93.2 (was 63.2) |

- Single-user TPOT is 1.5% slower: standalone decode is unchanged (17.50 ms
  against 17.54 ms), so the cost is host work for six KV groups instead of
  one, including the plugin padding the global group's table every step.
- Shared 6,144-token prefix plus 256 new tokens, 1 user: 323 ms median TTFT
  once the prefix is cached, against about 4.5 s to prefill 6.4K tokens.
  An 8K-token chat turn's follow-up: 1.31-1.69 s with the cache, 7.4-7.8 s
  without, same 48 tokens.
- Standalone traced decode: batch 1 17.50 ms, batch 2 28.92 ms, batch 4
  35.42 ms (was 56.49 ms; kernel time 50.9 -> 31.5 ms per step).

## Known limits and next steps

- Device sampling would remove the 1.2-2.8 ms logits readback and enable
  async decode, but needs a custom top-k kernel.
- Batch 2-4 decode still runs generic attention, norms and `topk_router_gpt`
  (68 us per layer against 36 us for the batch-1 router op).
- A prompt length never seen before compiles new programs once (1-20 s on a
  cold JIT cache); the disk cache keeps them across restarts.
- Prefill runs at about 1,400 tokens/s; an 8K first turn still takes 5 s.

## Reproduce

    pixi run python tests/smoke_chunked.py
    pixi run python tests/smoke_consistency.py --prompt-len 1100 --batch 3
    pixi run python scripts/bench_decode.py --batch 4 --steps 128 --trace
    pixi run serve-gemma4   # then, in another shell:
    pixi run python scripts/serve_check.py
    pixi run bash scripts/bench_tis.sh <label> c1-128-128-n8 c4-128-128-n16
    uv run --no-project --with-requirements scripts/gpqa_eval.requirements.txt \
        python scripts/gpqa_eval.py --count 40 --out results/quality/gpqa-<label>
