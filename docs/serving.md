# Serving Gemma 4 26B-A4B on N300 with vLLM

The vLLM backend in `gemma4_tt/vllm_model.py`: how it works, how it was
optimized and measured, and what is left. Stack: vLLM 0.32.0, vllm-tt-plugin
`0c12fd0`, transformers 5.16.1, ttnn 0.81.1+conda.8, one N300d (two Wormhole
chips, TP=2). The investigation that started this work is
`docs/gemma4-serving-investigation.md`; the earlier single-user decode work is
`docs/decode-optimization-report.md`. Two milestones are described here: the
serving backend (`feat/serving`) and the follow-up on compatibility, sampling,
quality and prefill (`feat/quality-sampling`); sections say which one a result
belongs to where it matters.

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
32K tokens (`GEMMA4_KV_POOL_GIB` overrides it). Weights take 8.0 of
11.87 GiB per chip; the rest holds prefill activations and the decode traces.

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

Every piece is padded to a fixed set of lengths (`prefill_rows` in
`gemma4_tt/tt/model.py`): whole tiles for a first piece of up to 512 tokens,
else multiples of 128. The programs of each length are built once, in
`warmup_model_prefill`, for every length up to the scheduler's chunk size
(36 s at 2,048 tokens). A prompt length the server has not seen therefore
reuses its bucket's programs; only the KV fill (per 32 rows) and the expert
chunk counts (per routing) can still build small programs later (see Prefill
length buckets). The fill page tables keep their full width, so a new chunk
offset builds nothing.

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
other backends either. Both paths are equally accurate against fp32 (KL 0.457
at batch 1, 0.423 at batch 4, see Model quality), but each is about 0.44 KL
from fp32, so on flat distributions (code continuations, raw completions
without the chat template) the two can disagree within a few tokens.

### Sampling

The model-side sampler (`sample` in `gemma4_tt/vllm_model.py`) runs on the
host after the logits are read back: greedy is an argmax, sampling takes the
top k <= 1,024 candidates and applies temperature, top-p and a seed keyed by
(seed, position). The final softcap (30 tanh(x / 30)) keeps the order, so the
decode step returns the logits before it and only the k candidates are
capped; logprobs and vLLM's own host sampler (penalties, min-p, larger top-k)
still get the whole capped row. Sampling on the device and the plugin's async
decode were evaluated and not built (see below).

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

### Expert-major prefill in one-tile chunks (`feat/quality-sampling`)

Prefill groups every expert's (token, slot) pairs into chunks and runs one
indexed sparse matmul pair over them (`_experts_prefill` in
`gemma4_tt/tt/moe.py`). A tt-scope capture of a 2,048-token prefill (see
Profiling) showed the sparse matmuls at 33% of the time and the glue around
them (row gathers, slices, concats, untilize) at about as much again. With
128-row chunks, 224 chunks held 16,384 real pairs: 43% of the matmul rows were
padding. Changes:

- chunks are always one tile (32 rows), which cuts the padding rows; the
  matmul time follows the rows (2,048 tokens: 1.28 s with 32-row chunks,
  1.34 s with 64, 1.54 s with 128);
- each sparse matmul call (at most 128 chunks) gathers its own rows instead
  of slicing one big gather;
- padding rows gather token 0 with weight 0, which zeroes their result, so the
  appended zero row and its concat are gone.

Kernel time per chip for a 2,048-token prefill fell from 1,390 to 1,110 ms.
The logits are unchanged to PCC 1.00000 and the prefill KL against fp32 is
0.264 (earlier runs 0.20-0.33).

### Prefill length buckets (`feat/quality-sampling`)

Before, every new prompt length built new programs on its first prefill
(0.6 s with cached kernel binaries) and sometimes compiled kernels. On a fresh
server, eight chat prompts at new lengths took 3.4-4.9 s on average with
single requests up to 9.7 s; the same lengths took 1.39 s once built. Pieces
now pad to the lengths in Architecture, warmed at startup. 256-row buckets
were tried first and cost a 144-token prompt 56 ms of TTFT (238 -> 294 ms),
which led to the whole-tile buckets for short first pieces.

Two shapes still vary inside a bucket. The KV fill writes only up to the
last live tile, so its few ops build per 32-row length; that measured as
nothing (300, 333 and 365 tokens in one bucket: 0.46, 0.38, 0.39 s on first
use). The chunk count of the expert matmuls depends on the routing (rounded
to 16 chunks), so the first request whose routing needs a new count builds
that program (about 0.4 s). A long-running server collects them
quickly; padding to the worst-case count would cost every request compute.

### Lazy softcap (`feat/quality-sampling`)

Capping 262,144 logits on the host took 0.34 ms per row (1.35 ms at batch 4)
on every step; greedy and top-k sampling now cap only their candidates (see
Sampling). Serving TPOT: 18.46 -> 18.14 ms at 1 user, 37.75 -> 36.44 ms at 4.

### Smaller changes

- **Decode interleave.** An 8K prompt arriving during a stream was measured
  with the plugin's settings (see Measurements); one decode step per prefill
  chunk halves the stream gap at the same TTFT.
- **Compile variety.** Expert-major prefill rounds its chunk count to 16,
  since each new count compiled new programs.
- **Not kept.**
  - Reading the logits into a kept host tensor (`copy_device_to_host_tensor`)
    and converting its shards: 1.0 ms against 0.66 ms for to_torch's mesh
    composer, right after the trace ran.
  - Two Ethernet links for the prefill all-gathers (`num_links=2`): the
    gather hung (the fabric runs one link between the N300's chips), which
    needed a card reset.
  - Untilizing each expert call's output and concatenating row-major instead
    of a tile concat: no change.
  - BFP8 experts in a few layers: see Model quality.

## Device sampling and async decode (evaluated, not built)

The plugin's async decode overlaps the host work of one step with the device
work of the next. It needs the model to sample on the device, write the token
back into the decode trace's input and advance the position there
(`docs/DECODE_RELOAD_CONTRACT.md` in vllm-tt-plugin). A standalone split of
one traced step (a one-off script, not kept; 30 steps, median):

| Part | Batch 1 | Batch 4 |
|---|---:|---:|
| Host inputs (embedding, RMSNorm, RoPE tables) | 0.46 ms | 0.72 ms |
| Input copies | 0.67-0.80 ms | 0.71 ms |
| Trace execution | 15.7 ms | 31.0 ms |
| Logit readback and softcap | 1.19-1.40 ms | 2.5-2.7 ms |

Serving TPOT is 18.1 ms at batch 1, so async decode could hide about 2.4 ms
of host work per step, minus the cost of sampling on the device:

- `ttnn.topk` has a cliff: k = 32 over 8,192 columns takes 0.11 ms, 16,384
  0.17 ms, 32,768 0.29 ms, but 65,536 16.7 ms (k = 64: 31.7 ms); a chip's
  131,072 logits as 4 x 32,768 or the LM head's 16 x 8,192 pieces cost
  1.2-1.8 ms before the cross-chip gather and the final draw.
  tt-transformers' `TTSampling` splits the vocabulary only on a single
  device, and upstream's Gemma 4 turns device sampling off above 64K
  columns per chip, which is this case.
- `ttnn.sampling` takes BF16 candidate logits (greedy near-ties would differ
  from the fp32 argmax), at most 32 users, and its own seed scheme.
- The token feedback needs the embedding on the device (0.74 GB per chip from
  the vocabulary-split table, out of the KV pool) and the batch-1 compact FP32
  inputs (RMSNorm of the embedding, RoPE tables) built in the trace.

Net: about 0.6-1 ms per step (4%) at batch 1 and about 10% at batch 4, for a
large change with weaker sampling semantics. The lazy softcap took the cheap
part (0.34 / 1.35 ms).

MTP speculative decoding was evaluated on paper. This checkpoint has no MTP
weights; upstream's drafter is a separate "assistant" model (4 layers) that
reads the target's KV cache. Verifying K drafts here needs either a multi-token
decode or a prefill that starts off the 128-token block grid, which the
prefix-cache invariant forbids. On this DRAM-bound MoE, verifying 4 tokens
reads close to 4x the expert weights (a batch-4 step costs 35 ms against
17.5 ms at batch 1), so the best case is about 15-20% at batch 1.

## Model quality and precision (`feat/quality-sampling`)

**The QAT weights are Q4_0, which BFP4 cannot hold.** Every 32-input block of
the expert, MLP and attention weights is an integer multiple of one step
(4-bit integers -8..7, roughly Gaussian, 6% of the blocks use -8), stored in
BF16. The step is learned (it does not follow llama.cpp's max / -8). The
device stores the experts as BFP4: blocks of 16 along the output dimension,
a shared power-of-two exponent and a sign plus 3-bit magnitude, so 15 levels
on a power-of-two grid. Weight error of layer 3's experts (relative MSE):

| Format | SNR |
|---|---:|
| BFP4 as stored now (blocks along outputs) | 20.2 dB |
| BFP4 with the best exponent per block | 20.4 dB |
| BFP4 with blocks along inputs, as Q4_0's | 20.4-20.8 dB |
| Per-output-channel rescaling before BFP4 | 18.9-20.0 dB |
| BFP8 | 43.2 dB |

No BFP4 layout or rounding recovers the grid: a power-of-two step with 15
levels cannot land on a learned step's 16 levels, and every variant stays near
20 dB. Exact weights need a kernel that dequantizes Q4_0 blocks itself; a Q4_0
block (32 values and an FP16 step) takes 4.5 bits per value, as a BFP4 tile
does, so the DRAM traffic stays the same. The decode streams are custom
kernels and could take it, but prefill uses ttnn's sparse matmul.

What it costs, from a CPU emulation of the fp32 reference with BFP4 experts
(bit-exact torch BFP4, 2,048 README tokens): KL 0.197 against fp32, top-1
87.4%; at 512 tokens 0.341. The BF16 reference alone is at 0.163 at 512 tokens,
so BFP4 experts are the largest single loss; the device model is at 0.42-0.46
(decode) and 0.26 (prefill, 2,048 tokens).

**Batch paths.** `scripts/compare.py --batch 4` scores the batched decode path
on the same teacher-forced text: KL 0.423 and top-1 77.2% against 0.457 and
75.8% for batch 1 (3 perturbed seeds each, `results/accuracy/batch-path/`), so
concurrency changes which kernels run, not the accuracy. At a flat position
the two paths can still differ from each other by as much (KL 0.50 between
the batch-1 and batch-4 logits of a 3,500-token code prompt).

**Mixed precision was not kept.** The KV pool needs 1.3 GiB per chip for four
32K requests and has 2.5, so about five layers' experts fit in BFP8 (0.2 GB
per chip each), at about 0.27 ms more per decode step and layer (+8% TPOT for
five). In emulation, exact experts in layers 0-5 moved KL 0.197 -> 0.189 and
in layers 2, 7, ..., 27 to 0.218 (2,048 tokens); a single exact layer moved it
by -0.02 to +0.08 at 512 tokens. All of these are inside the run-to-run noise
(0.04-0.07), so no layer choice is supported by the data.

**Math fidelity** stays LoFi for the BFP4 experts: their weights carry 3
mantissa bits, and in the earlier precision study (`results/accuracy/precision_study.json`)
LoFi and HiFi2 runs differed only within the noise.
Accumulation is FP32 in the destination registers, with few K blocks in the
prefill sparse matmuls because each block boundary rounds the partial sum to
BF16.

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

`feat/quality-sampling` added two captures of one 2,048-token prefill of
random tokens, before and after the one-tile expert chunks:

    pixi run tt-scope tracy scripts/profile_workload.py --name prefill2048 --programs 12000 -- --prompt-len 2048 --decode-steps 0

The per-type tables are `results/profile/prefill-2048-{before,after}.txt`.
Findings that shaped decisions:

- before: 2,569 ops, 1,390 ms kernel time on chip 0, 157 ms of gaps. Sparse
  matmuls 462 ms (33%), all-gathers 161 ms, the expert row gathers
  (`ttnn.embedding`) 153 ms, slices 81 ms, GELU 69 ms, untilize 65 ms,
  concat 60 ms. Per layer, the glue around the experts cost about as much as
  the expert matmuls (15 vs 16 ms), and the gate|up sparse matmul ran on only
  24 cores (its 1D config splits the 768 output columns, one tile per core).
- after: 3,499 ops, 1,110 ms. Sparse matmuls 368 ms, gathers 80 ms, slices
  30 ms. The all-gathers (162 ms, about 6 GB/s over the one Ethernet link)
  are now the second item.

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
- `feat/quality-sampling`: a one-off host split of a traced step (input build,
  copies, execution, readback; the table under Device sampling), `ttnn.topk`
  and all-gather microbenchmarks, standalone prefill timing on README text,
  `scripts/ttft_lengths.py` against the server, `scripts/compare.py --batch`
  for the batched path's accuracy, and a CPU emulation of BFP4 experts on the
  fp32 reference (no ttnn: a ttnn host conversion starts the device cluster,
  and doing that next to a device job once needed a card reset).

## Measurements

Serving: `scripts/bench_tis.sh` (tt-inference-server's client recipe: chat
endpoint, random prompts, ignore-EOS, greedy), mean of 3 runs after a
discarded warm-up, CPU governor `performance`. Round 2:
`results/compare/tis-match/`. `feat/serving`: `results/compare/feat-serving/`.
`feat/quality-sampling`: `results/compare/quality-sampling/`.

| Point | TTFT round 2 / serving / now | TPOT round 2 / serving / now | Output tok/s now |
|---|---:|---:|---:|
| 1 user, 128 in / 128 out | 237 / 238 / 236 ms | 18.17 / 18.45 / 18.08 ms | 50.5 |
| 1 user, 2,048 / 128 | 1.57 / 1.48 / 1.21 s | 19.32 / 19.67 / 19.33 ms | 34.9 |
| 1 user, 8,192 / 128 | 6.95 / 5.15 / 4.36 s | 19.69 / 20.02 / 19.65 ms | 18.7 |
| 2 users, 128 / 128 | - / 308 ms / - | ~42 / 30.74 ms / - | - |
| 4 users, 128 / 128 | 736 / 694 / 695 ms | 57.95 / 37.75 / 36.19 ms | 96.7 (63.2, 93.2) |

Sampled (generation-config temperature 1, top-k 64, top-p 0.95), measured after
the lazy softcap and before the prefill changes: TPOT 18.50 ms at 1 user,
37.34 ms at 4 users.

Standalone prefill of README text (`feat/serving` -> now, one call, min of 2):
512 tokens 0.514 -> 0.450 s, 2,048 1.581 -> 1.269 s, 4,096 2.640 -> 2.405 s,
8,192 5.618 -> 5.101 s. One-token requests at lengths the server has not seen
(`scripts/ttft_lengths.py`, whole-request time, `results/compare/quality-sampling/ttft-unseen-lengths.txt`):
3.4-4.9 s mean on a fresh `feat/serving` server (single requests up to 9.7 s,
1.39 s once built), 1.17-1.37 s now.

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

Before chunked prefill the same arrival held the stream for 5.66 s. With the
faster prefill (`serve_check.py` on `feat/quality-sampling`): longest gap
1.45 s, the 8K request's first token after 5.4 s, and the uncached 8K
follow-up turn 6.2-6.6 s (cached: 1.26 s).

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
- `scripts/serve_check.py preempt` (`feat/quality-sampling`), on a server
  started with `GEMMA4_KV_POOL_GIB=0.2 ... --max-model-len 4096`: four number
  lists outgrow the 313-block pool together, vLLM preempts one and resumes it
  by recomputation, and all four equal their solo runs over 1,536 tokens
  (twice, one preemption each). Code and raw-completion prompts were first
  tried and dropped: their flat distributions diverge within a few tokens from
  batch numerics alone, which hides what the check is for.
- A feature probe against the server (`feat/quality-sampling`): tool calls
  with `tool_choice=auto` and a tool-result turn, JSON schema and JSON object
  outputs, regex and choice constraints (enforced: a choice of
  "purple"/"seven" for a yes/no question returned "seven"), reasoning
  output, logprobs with top logprobs, presence/frequency/repetition
  penalties, min-p with top-k 5,000, `n=2`, stop strings with `min_tokens`,
  `logit_bias`, `echo`. All behave as on vLLM's other backends.

## Capabilities

| Capability | Status |
|---|---|
| Continuous batching, 1-4 concurrent requests | Supported; more than 4 rows is refused at load |
| Prefix caching | Supported, 128-token hit granularity |
| Chunked prefill with decode interleave | Supported |
| Preemption and resumption | Supported, verified with a small pool (`serve_check.py preempt`) |
| Streaming, stop sequences, EOS, cancellation | Supported |
| Temperature, top-k <= 1,024, top-p, seed | Model-side sampler on the host; a seeded request can draw differently when a co-batched request forces vLLM's host sampler |
| Penalties, min-p, larger top-k, logprobs | vLLM's host sampler (per-step fallback) |
| Structured output, tool calls, reasoning parser | Through vLLM (host grammar, gemma4 parsers) |
| `tool_choice="required"` or a named tool | Not enforced: vLLM 0.32's gemma4 tool parser skips the JSON constraint for them (the model answers as for `auto`) and has no structural tag; the same on every backend |
| `n > 1`, `logit_bias`, `echo` | Supported |
| Embeddings endpoint | No (generative model only) |
| `prompt_logprobs` | No (refused by the TT plugin) |
| Async decode, device sampling, penalties or grammar | No: evaluated, about 4% at batch 1 (see Device sampling and async decode) |
| vLLM Python streaming-input sessions | No: can resume prefill off the block grid |
| Multimodal, speculative decoding, data parallel | No (MTP: no heads in this checkpoint, see above) |
| KV pool size | `GEMMA4_KV_POOL_GIB` (default 2.5 GiB per chip) |

## Future work

- **Quality: a Q4_0 expert kernel.** The largest accuracy loss is BFP4's
  representation of the QAT weights (Model quality). The decode streams could
  read Q4_0 blocks (4.5 bits per value) and dequantize in the kernel; prefill
  would need the same in its expert matmuls.
- **Device sampling and async decode.** Worth about 4% at batch 1 and 10% at
  batch 4 with today's ops; a fused LM-head top-k (candidates per 8,192-column
  piece, written by the matmul's own cores) would make it pay more.
- **Batch 2-4 decode.** The router (`topk_router_gpt`, 68 us per layer
  against 36 us for the batch-1 router op, about 1 ms per step), 151 norm ops
  per step and generic attention now lead; the batch-1 compact boundaries
  could extend to 4 rows. Expert slots shared between users are read twice.
- **Prefill.** About 1,700 tokens/s at 2,048. The all-gathers (15%) are
  link-bound on one Ethernet link; the gate|up sparse matmul uses 24 of 64
  cores; the expert row gathers and the gather back still cost 6 ms per layer.
- **Tool choice.** A structural tag for Gemma 4's native tool-call syntax in
  vLLM would make `required` and named tool choice enforceable.
- **Validation.** More GPQA questions or seeds before comparing precision
  choices.

## Reproduce

    pixi run python tests/smoke_chunked.py
    pixi run python tests/smoke_consistency.py --prompt-len 1100 --batch 3
    pixi run python scripts/bench_decode.py --batch 4 --steps 128 --trace
    pixi run serve-gemma4   # then, in another shell:
    pixi run python scripts/serve_check.py
    pixi run python scripts/ttft_lengths.py 3
    pixi run bash scripts/bench_tis.sh <label> c1-128-128-n8 c4-128-128-n16
    GEMMA4_KV_POOL_GIB=0.2 pixi run serve-gemma4 --max-model-len 4096   # then:
    pixi run python scripts/serve_check.py preempt
    pixi run python scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb 1 --batch 4
    uv run --no-project --with-requirements scripts/gpqa_eval.requirements.txt \
        python scripts/gpqa_eval.py --count 40 --out results/quality/gpqa-<label>
