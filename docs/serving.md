# Serving Gemma 4 26B-A4B on N300 with vLLM

The vLLM backend in `gemma4_tt/vllm_model.py`: how it works, how it was
optimized and measured, and what is left. Stack: vLLM 0.32.0, vllm-tt-plugin
`0c12fd0`, transformers 5.16.1, ttnn 0.81.1+conda.8, one N300d (two Wormhole
chips, TP=2). The investigation that started this work is
`docs/gemma4-serving-investigation.md`; the earlier single-user decode work is
`docs/decode-optimization-report.md`.

    pixi run serve-gemma4                       # 127.0.0.1:8000, model "gemma4"
    pixi run python scripts/serve_check.py      # request lifecycle checks

## Architecture

### KV cache: vLLM owns every layer

Before, vLLM paged only the 5 global layers, and the adapter kept a private
1,024-token ring per batch slot for the 25 sliding layers. That ring state
blocked prefix caching, chunked prefill and preemption.

Now `get_kv_cache_spec` declares the sliding layers as `SlidingWindowSpec` and
the global layers as `FullAttentionSpec`. vLLM's hybrid manager forms 5
sliding groups and 1 global group, each with its own page table, and frees
sliding blocks that leave the window. A block id belongs to one group at a
time, so the i-th layer of every group shares one K and one V buffer
(`gemma4_tt/tt/kv_cache.py`). To keep pages equal, vLLM gives the global
layers 128-token blocks (1 KV head of 512 per chip) next to 64-token sliding
blocks (4 heads of 256). Each layer reads the shared buffer through its own
geometry, which ttnn's paged ops accept as an override. The model holds no
per-request state, so preemption, resumption and batch-row moves need no
adapter hooks (`release_request` and `slot_remap` are gone).

The pool is `KV_POOL_BYTES` = 2.5 GiB per chip: 3,860 blocks, 7.7 requests of
32K tokens. Weights take 8.0 of 11.87 GiB per chip; the rest holds prefill
activations and the decode traces.

### Prefix caching and chunked prefill

`Gemma4TT.prefill(tokens, start, tables)` computes positions
`[start, len(tokens))` and writes only their K/V. A chunk at position 0
attends over its own K/V; a later chunk uses chunked SDPA over the paged
cache, which by then holds the earlier positions and the chunk itself. The
chunk start goes to the device as a tensor, so the programs no longer depend
on the offset: with the start as an integer, every new offset compiled new
programs (2.3 s on a cold JIT cache); now a 53-token chunk at a new offset
takes 0.22-0.25 s after the first one.

`start` must lie on the block grid, the lcm of the group block sizes (128
tokens with `--block-size 64`). vLLM schedules on that grid, so prefix hits
and the plugin's chunk ends fall on it, and every group then writes from the
start of a block of its own: shared prefix blocks are never rewritten. The
adapter rejects a `prefill_chunk_alignment` off the grid at startup. Fills
stop at the last tile holding a real position, so padding never reaches a
block vLLM did not allocate. Prompts longer than 4,096 tokens run in pieces
to bound activation memory.

Prefix caching is vLLM's automatic prefix caching with a 128-token hit
granularity. Chunked prefill uses the plugin's scheduler with
`--max-num-batched-tokens 2048` and one decode step after every prefill chunk.

### Multi-user decode

Decode captures one metal trace per active row count (1-4).

- **1 row**: the fused path from the earlier decode work (compact FP32
  boundaries, fused tails, router op, expert and MLP streams). Unchanged.
- **2-4 rows**: generic attention and norms, but the routed experts and the
  dense MLP now run through the same DRAM-streaming kernels as batch 1
  (`gemma4_tt/tt/expert_stream.py`). One op pair covers all `8 x B` expert
  slots (see below).

Batch 1 and batches 2-4 run different kernels, which round differently.
Which one computes a step depends on what else is running, so a greedy
request can take a different branch at a near-tie (observed once in five
concurrent rounds, at token 20). vLLM does not promise batch invariance on
other backends either.

## Optimization methods

### Trace safety (correctness)

The investigation saw identical raw-token requests disagree from their first
token. A standalone repeat was bitwise stable, so the difference had to come
from serving. Prefill and decode inputs dumped from the server, replayed
standalone in the server's call order, reproduced it: a buffer allocated
after a decode trace capture can overlap the trace's intermediates, and every
replay overwrites it. Two such buffers existed:

- **Decode trace inputs and lazy constants.** Each batch size's inputs were
  allocated just before its own capture, and some constants (rotate_half's
  matrix, MoE masks) on first use. Page tables are skipped when unchanged, so
  a replay could use corrupted block ids, which hung the card.
  `_prepare_traces` now allocates every batch size's inputs and runs every
  batch size once before the first capture.
- **`ttnn.all_reduce` in prefill.** It caches 12 KB of L1 semaphores per new
  input shape. The first prefill of a new length after capture cached them in
  trace memory, and the next prefill of that length returned garbage (logits
  off by 28). Prefill now uses decode's reduction: `all_gather_async` on
  semaphores allocated at load, plus a local sum.

Both bugs predate this work; each fix was verified on the old code too.

### Batched expert and MLP streams

Profiling a batch-4 step (next section) showed the routed experts at 26 of
51 ms: indexed `sparse_matmul` read the 32 expert slots at about 85 GB/s,
plus slicing and routing glue. The batch-1 stream kernels read at about
190 GB/s and already multiply whole 32-row tiles, so they were extended to
B rows instead of writing new kernels:

- slot `s = 8 t + j` is token `t`'s j-th expert;
- the gate|up writer builds each slot's score tile with the routing weight in
  that token's row and zeros in the other token rows;
- the down kernel's existing slot sum therefore leaves each token's experts
  in its own row; it now streams activations slot by slot, so its buffer
  stays two slots deep at any batch.

The dense MLP reuses the same kernels with every slot serving every row.

### Smaller changes

- **Decode interleave.** An 8K prompt arriving during a stream was measured
  with the plugin's settings (see Measurements); one decode step per prefill
  chunk halves the stream gap at the same TTFT.
- **Compile variety.** Expert-major prefill rounds its chunk count to 16,
  since each new count compiled new programs.
- **Not kept.** On-device top-k for sampling: `ttnn.topk` over a chip's
  131,072 logits takes 38-71 ms (standalone microbenchmark), far more than the
  1.2-2.8 ms logit readback it would remove.

## Profiling

**tt-scope** (v0.4.0, `tracy` mode) was used for two captures of one traced
batch-4 decode step, before and after the batched streams:

    pixi run tt-scope tracy scripts/profile_workload.py --programs 8000 -- --batch 4 --decode-steps 2 --trace
    pixi run python scripts/trace_breakdown.py artifacts/runs/<run> --phase decode1

`scripts/trace_breakdown.py` recomputes op time and gaps from device firmware
cycles; a per-op grouping of its data gave the tables in
`results/profile/decode-b4-{before,after}.txt` (raw runs stay in the
git-excluded `artifacts/runs/`). The "before" capture ran the KV rewrite just
before the streams, not the round-2 code. Findings that shaped decisions:

- before: 2,348 ops, 50.9 ms kernel time per chip, 0.86 ms of gaps. Expert
  `sparse_matmul` 20.5 ms, gate/up slicing 3.3 ms, `topk_router_gpt` 2.1 ms.
  Gaps were small, so the fix had to cut kernel time, not dispatch.
- after: 1,898 ops, 31.5 ms. Expert gate|up 191 us and down 86 us per layer
  (about 204 and 226 GB/s, close to the DRAM limit for 32 slots); the router,
  norms and LM head now lead.

**Other tools, not tt-scope:**

- standalone timing: `scripts/bench_decode.py --trace` (per-step wall time),
  `scripts/trace_time.py` (blocking trace time), a one-off script that split a
  step into input copies, trace execution and logit readback (inputs 1.5 ms,
  readback 1.15 ms at batch 1 and 2.83 ms at batch 4);
- `ttnn.get_memory_view` (`scripts/memory_report.py`): per-chip DRAM after
  weights, and the leak test that found `ttnn.all_reduce`'s per-shape cache;
- server logs with `GEMMA4_TIMING=1`: per-chunk prefill time and per-step
  decode time, used for the interleave and compile findings;
- `vllm bench serve` (`scripts/bench_tis.sh`) and `scripts/serve_check.py`.

## Measurements

Serving: `scripts/bench_tis.sh` (tt-inference-server's client recipe: chat
endpoint, random prompts, ignore-EOS, greedy), mean of 3 runs after a
discarded warm-up, CPU governor `performance`. Before: round-2 code,
`results/compare/tis-match/`. After: `results/compare/feat-serving/`.

| Point | TTFT before | TTFT after | TPOT before | TPOT after | Output tok/s after |
|---|---:|---:|---:|---:|---:|
| 1 user, 128 in / 128 out | 237 ms | 238 ms | 18.17 ms | 18.45 ms | 49.5 |
| 1 user, 2,048 / 128 | 1.57 s | 1.48 s | 19.32 ms | 19.67 ms | 32.2 |
| 1 user, 8,192 / 128 | 6.95 s | 5.15 s | 19.69 ms | 20.02 ms | 16.6 |
| 2 users, 128 / 128 | - | 308 ms | ~42 ms (investigation) | 30.74 ms | 60.8 |
| 4 users, 128 / 128 | 736 ms | 694 ms | 57.95 ms | 37.75 ms | 93.2 (was 63.2) |

Prefix reuse (`results/compare/feat-serving/c1-prefix6144-*`, 6,144 shared
plus 256 new tokens, 1 user): 323 ms median TTFT once the prefix is cached.
`serve_check.py`'s 8K-token chat follow-up: 1.29-1.69 s with the cache,
7.36-7.76 s with a cache salt, same 48 tokens.

Decode interleave, 8,192-token prompt arriving during a stream (server logs
and `serve_check.py arrival`, steady state, 3 runs each; numbers in commit
`1941c2f`):

| Chunk, prefill steps per decode step | Longest stream gap | 8K TTFT |
|---|---:|---:|
| 2,048, 2 (plugin default) | 3.38 s | 6.53 s |
| 2,048, 1 (configured) | 1.72 s | 6.56 s |
| 1,024, 1 | 1.00 s | 7.18 s |

Before chunked prefill the same arrival held the stream for 5.66 s.

Standalone traced decode (`scripts/bench_decode.py --trace`, 128 steps,
median): batch 1 17.54 -> 17.50 ms, batch 4 56.49 -> 35.42 ms (70.8 -> 112.9
tok/s total), batch 2 28.92 ms.

Uncertainty:

- Single-user serving TPOT is 1.5% slower while standalone decode is
  unchanged. The likely cause is host bookkeeping for six KV groups instead of
  one (the plugin pads the global group's table every step); it was not
  profiled.
- Decode accuracy against the fp32 reference, 3 seeds with matched noise: KL
  0.421 (round 2) vs 0.457 (now), top-1 0.777 vs 0.758. A bisect puts all of
  it on the prefill reduction, the more exact of the two; prefill-only KL
  moved the other way. This is within the known ulp sensitivity, but 3 seeds
  cannot exclude a small effect (`results/accuracy/matched-noise/README.md`).
- GPQA Diamond, upstream QB2 settings: 26/40, 7/10 on upstream's subset
  (`results/quality/gpqa-feat-serving/`). Upstream gates its higher-precision
  QB2 build on 8/10; with 40 sampled answers the 95% interval is about +-15
  points, so this is a local baseline only.

## Validation

- `tests/smoke_chunked.py` (random weights): chunked prefill, a prefix hit on
  shared blocks, vLLM-shaped tables, the null block never written, an
  off-grid start refused. PCC >= 0.99998 against one full prefill.
- `tests/smoke_consistency.py`, batches 1-4 and past the window: decode vs
  prefill PCC 0.99996-0.99999. `tests/smoke_inactive.py`: 0.99998.
- `scripts/bench_expert_stream.py --rows 4`: batched stream against the
  prefill expert path, PCC 0.99972-0.99986 per row.
- `scripts/greedy_check.py`: 4/4 prompts match the round-2 greedy tokens.
- `scripts/serve_check.py` on the final code: concurrent requests (first 16
  tokens, see Multi-user decode), raw tokens at temperature 0 twice, an 8K
  arrival during a stream, a cancelled stream, a prefix hit confirmed through
  vLLM's metrics against a salted miss, seeded sampling. All pass.
- Not exercised: preemption and resumption under KV pressure (supported by
  design, unverified).

## Capabilities

| Capability | Status |
|---|---|
| Continuous batching, 1-4 concurrent requests | Supported; more than 4 rows is refused at load |
| Prefix caching | Supported, 128-token hit granularity |
| Chunked prefill with decode interleave | Supported |
| Preemption and resumption | By design, unverified |
| Streaming, stop sequences, EOS, cancellation | Supported |
| Temperature, top-k <= 1,024, top-p, seed | Model-side sampler on the host; a seeded request can draw differently when a co-batched request forces vLLM's host sampler |
| Penalties, min-p, larger top-k, logprobs | vLLM's host sampler (per-step fallback) |
| Structured output, tool calls, reasoning parser | Through vLLM (host grammar, gemma4 parsers) |
| `prompt_logprobs` | No (refused by the TT plugin) |
| Async decode, device sampling, penalties or grammar | No: needs tokens sampled on the device |
| vLLM Python streaming-input sessions | No: can resume prefill off the block grid |
| Multimodal, speculative decoding, data parallel | No |

## Future work

- **Device sampling.** A custom top-k kernel over the vocabulary would remove
  the logit readback (1.2-2.8 ms per step) and allow the plugin's async
  decode, which overlaps host work with the device.
- **Batch 2-4 decode.** The router (`topk_router_gpt`, 68 us per layer
  against 36 us for the batch-1 router op), 151 norm ops per step and generic
  attention now lead; the batch-1 compact boundaries could extend to 4 rows.
  Expert slots shared between users are read twice.
- **Prefill.** About 1,400 tokens/s; an 8K first turn still takes 5 s. A
  never-seen prompt length compiles new programs once (1-20 s cold); a fixed
  set of padded lengths warmed at startup would bound that.
- **Validation.** A forced-preemption serving test, and more GPQA questions or
  seeds before comparing precision choices.

## Reproduce

    pixi run python tests/smoke_chunked.py
    pixi run python tests/smoke_consistency.py --prompt-len 1100 --batch 3
    pixi run python scripts/bench_decode.py --batch 4 --steps 128 --trace
    pixi run serve-gemma4   # then, in another shell:
    pixi run python scripts/serve_check.py
    pixi run bash scripts/bench_tis.sh <label> c1-128-128-n8 c4-128-128-n16
    uv run --no-project --with-requirements scripts/gpqa_eval.requirements.txt \
        python scripts/gpqa_eval.py --count 40 --out results/quality/gpqa-<label>
