# Gemma 4 26B-A4B on N300: single-user decode, round 2

Baseline is `5887694` (start of this round), final is `f158df1`. Both use the
QAT checkpoint with BFP4 experts and BFP8 elsewhere on one N300d (1x2 mesh),
ttnn 0.81.1+conda.8, profiled with tt-scope v0.4.0 (`e9f5941`).

Host for every number below: power profile `performance`, governor `performance`
on all 20 CPUs (EPP default). Baseline and final were measured back to back under
this setting. Earlier numbers in commit messages were taken under `powersave` /
`balanced` and are not directly comparable for host-side times.

## Results

| Level | Baseline | Final |
|---|---:|---:|
| Device: traced step, kernel time per chip (profile) | 31.6 ms, 2,199 ops | 15.0 ms, 574 ops |
| Device: blocking trace execution, batch 1 | 31.47 ms | 15.63 ms |
| Standalone decode, batch 1 (`bench_decode.py --trace`) | 33.3 ms (30.0 tok/s) | 17.5 ms (57.0 tok/s) |
| Standalone decode, batch 4 | 55.5 ms (72.1 tok/s total) | 56.5 ms (70.8 tok/s total) |
| vLLM 1 user, 128 in / 128 out: TPOT | 35.2 ms | 18.7 ms (53.4 tok/s) |
| vLLM 1 user, 2048 / 128: TPOT, TTFT | 36.7 ms, 1.47 s | 19.9 ms, 1.53 s |
| vLLM 4 users, 512 / 256: TPOT, output tok/s | 58.2 ms, 61.7 | 58.8 ms, 60.9 |
| Prefill 2048 tokens, standalone | 1.455 s | 1.509 s |

Files: `results/final/` (standalone, trace breakdowns), `results/bench/final-base`
and `results/bench/final` (vLLM, `vllm bench serve` format).

All the work targets batch 1. Batch 4 runs the old path and pays ~2% for the
padded down weights the streams need; TTFT pays ~4% for the same reason
(prefill slices the padding off).

## Correctness

- Decode 4 segments x 128 steps teacher-forced against the fp32 HF reference,
  6 seeds (seed 0 unperturbed, 1-5 with 2^-9 embedding noise; one run moves
  0.04-0.07 KL): baseline top-1 77.7%, KL 0.387, top-5 overlap 3.58; final 78.3%,
  0.368, 3.60. Files `results/accuracy/decode4x128-{wo-mlp,base-r2,final}-p*.json`.
  The FP32 residual stream and the more exact router (below) likely explain the
  small improvement.
- `tests/smoke_consistency.py`: decode vs prefill PCC 0.99998 (b1), 0.99999 (b4),
  0.99998 (1100-token sliding ring).
- `scripts/greedy_check.py`: 4 chat prompts match the baseline's greedy tokens up
  to the end-of-turn token.
- vLLM: `scripts/vllm_consistency.py` PASS (4 concurrent greedy requests equal
  the same requests alone).
- `scripts/soak.py`: 3,000 batch-1 and 1,000 batch-4 traced steps, device DRAM
  8.318 -> 8.318 and 8.327 -> 8.327 GiB per chip, no hangs.
- Per-op checks: `tests/boundary_check.py`, `tests/mlp_stream_check.py`,
  `tests/router_check.py`, `scripts/bench_expert_stream.py`.

## Early measurements the brief asked for

- **Gaps under trace:** 0.8 ms of device gaps per 34 ms step (0.37 us per op).
  The brief's 19 ms of gaps were host dispatch in eager mode. Under trace, small
  ops still cost 1.2-7 us of kernel time each, so op count mattered through
  kernel time, not gaps.
- **Cross-chip all-reduce of 2816 values:** `ttnn.all_reduce` 43 us on the
  tile-padded row (71 us for 5632); `all_gather_async` + local sum 34 us (58 us);
  the same gather of a compact 3-tile vector 10-14 us. Floor: 11 us for one tile.
  `all_reduce_async` and `num_links=2` hung the card.
- **Expert matmul bandwidth:** indexed `sparse_matmul` reads the 8 routed experts
  at ~75 GB/s (it splits only N, 24 cores). The bank-streaming kernels reach
  ~190 GB/s; ttnn's DRAM-sharded matmul reaches ~210 GB/s on the LM head.
- **LM head:** tied embedding, BFP8, vocab split over the chips (131,072 columns
  each); now 16 DRAM-sharded column splits per chip. **KV cache:** BFP8, paged
  DRAM; global layers use vLLM's page table, sliding layers a per-slot ring of
  1,024 tokens.

## What changed, in order

Blocking trace execution at batch 1 after each step (measured at commit time):

| Commit | Change | Trace exec |
|---|---|---:|
| `86e90f8` | LM head as 16 DRAM-sharded splits, untilize per split, softcap on host (numpy; torch's pool woke slowly) | 31.5 -> 30.9 ms |
| `2104d70` | Decode all-reduce = `all_gather_async` + `fast_reduce_nc` | -> 30.1 |
| `775e4ec` | **Expert streams** (generic ops): gate\|up reads each DRAM bank's (gate, up) pair blocks of the routed experts in 6 KB packets, GELU and routing weights fused; down sums the 8 slots in its K loop | -> 24.1 |
| `dab3331` | **Compact boundaries**: residual and partial sums as 3 dense FP32 tiles; pack op, 3-tile all-gather, one tail op per boundary for chip sum, all norms, residual adds, layer scalar | -> 19.7 |
| `350a68c` | Dense MLP through the same stream kernels | -> 18.65 |
| `3f435d9` | Fused q/k RMSNorm + RoPE (per-head sums as block-ones matmuls on compact tiles) | -> 17.7 |
| `fd5a079` | Router op: 11 cores of partial logits, semaphore reduce, top-8 and softmax on the reducer's RISC (71 -> 36 us; weight error vs fp32 0.0005, was 0.0069) | -> 16.7 |
| `b0767cc` | Tail reader pushes partials before weights; 6 tail cores | -> 16.5 |
| `b6147ae` | Host: sample only active rows, argmax for greedy, skip unchanged page tables, one FP32 input tensor (7 -> 3 trace input copies) | vLLM TPOT -1.5 ms |
| `15d07ee` | Router and MLP gate\|up as one merged program on disjoint cores | -> 15.66 |
| `f158df1` | Tails reduce RMS sums with an FPU scalar reduce | -> 15.6 |

## Ranked bottlenecks

Per-layer device time on chip 0 (mean of 30 layers, `results/final/trace-breakdown-*.txt`):

| Cost | Before (947 us/layer) | After (432 us/layer) |
|---|---:|---:|
| Expert matmuls + their glue | 181 + ~100 | 73 (2 stream ops) |
| Router | 69 | merged with MLP gate\|up: 51 for both |
| Dense matmuls: QKV / wo / MLP | 70 / 41 / 83 + 16 glue | 70 / 41 / 21 + (MLP gate\|up in the 51 above) |
| Norms, residual adds, conversions | ~150 | 61 (2 tail ops) |
| All-reduce | 125 | 28 (2 compact gathers) + 3 pack |
| q/k norm + RoPE | 60 | 24 |
| SDPA + cache update + head ops | 63 | 59 |
| LM head (per step) | 3.2 ms | 2.06 ms |

## Tried and not kept

- `all_reduce_async` and `num_links=2`: both hung the card (2 resets, logged).
- Packing a row into dense tiles with tile reshapes (47 us) or row-major
  all-reduce (no faster than tiles).
- Router as `ttnn.linear` + `ttnn.topk` + softmax: 161 us vs 72.
- K-split expert gate\|up in `sparse_matmul` (129 -> 98 us before the extra slice
  and add); padding the down to 48 cores: slower.
- Merging two DRAM-bound streams into one program: gate\|ups 19.1-20.4 ms, downs
  15.85 ms, both slower than back to back. Merging a latency-bound op with a
  DRAM-bound one (router + MLP gate\|up) did help.
- DRAM-sharded QKV fed directly by the tail: no gain (15.7 ms), reverted.
- RISC word copies for q/k RoPE partners: +0.4 ms.
- `paged_fused_update_cache`: no `cache_position_modulo`, so only the 5 global
  layers could use it (~0.3%).

The loop stopped after three consecutive attempts under 3% (down merge, DRAM-
sharded QKV, FPU reduce in the tails).

## Gap to the roofline

Weights per chip per layer are ~44 MB (QKV 12.3, wo 6.1, MLP 10.3, experts
13.8 with padding, router 0.7), a 182 us floor at 240 GB/s; the layer takes
432 us. The LM head floor is 1.63 ms; it takes 2.06 ms. Where the other ~7.8 ms
per step go:

- **Weight ops at 170-210 GB/s (~2.6 ms over floor):** QKV and wo are ttnn
  matmuls; the streams use 32x32 activation tiles (Wormhole has no `custom_mm`)
  and only 11 of 12 banks for the expert gate\|up.
- **Non-weight ops (~5.2 ms):** attention core 59 us (SDPA, two single-core
  cache updates, head split), tails 61 us, q/k RoPE 24 us, gathers 28 us per layer.
  Each generic op also pays a few us of launch and pipeline ramp.
- **Host (~1.9 ms standalone, ~3 ms in vLLM):** input copies, 1 MB logit
  readback, softcap and sampling. On-device top-k would remove most of it.
- **Chip 1 is 0.5 ms slower than chip 0** per step: stream readers are placed
  from chip 0's DRAM worker map, and the all-gathers make both chips wait for the
  slower one.

Next steps, in expected value: on-device sampling (host), fusing the tails into
the neighbouring ops (gpt-oss's fused decode does the all-reduce + norm inside the
next matmul), a custom attention block (cache update + SDPA), and extending the
streams to batch 4 for serving.

## Reproduce

    pixi run python scripts/bench_decode.py --batch 1 --steps 128 --trace
    pixi run python scripts/trace_time.py                       # blocking trace time
    pixi run bash scripts/final_standalone.sh <checkout> <label> # READ_TIME=$PWD/scripts/trace_time.py
    pixi run tt-scope tracy scripts/profile_workload.py --programs 8000 -- --decode-steps 3 --trace
    pixi run python scripts/trace_breakdown.py artifacts/runs/<run> --layer-ops 18 --head-ops 0
    pixi run python scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <seed>
    pixi run python tests/smoke_consistency.py --prompt-len 64 [--batch 4 | --prompt-len 1100]
    pixi run python scripts/greedy_check.py --compare results/greedy/base.json
    pixi run python scripts/soak.py
    pixi run serve-gemma4   # then: pixi run bash scripts/bench.sh <label>; pixi run python scripts/vllm_consistency.py

To run the baseline from a worktree, export `PYTHONPATH` inside the pixi
environment (`pixi run bash -c 'export PYTHONPATH=$PWD; ...'`): pixi's
activation otherwise points it back at the main checkout.
