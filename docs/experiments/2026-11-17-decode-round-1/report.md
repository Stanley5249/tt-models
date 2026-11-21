# Gemma 4 26B-A4B on N300: decode and serving optimization

Baseline is commit `5101e0e`, final is the commit that adds this file. Both run the same QAT checkpoint, BFP4 experts and BFP8 elsewhere, on one N300d (1x2 mesh).

## Results

vLLM serving (`scripts/bench.sh`, random prompts, server-default sampling). The baseline needed one fix to start at all (see "Bugs fixed").

| Scenario                | TTFT base -> final | TPOT base -> final | Output tok/s base -> final |
| ----------------------- | -----------------: | -----------------: | -------------------------: |
| 1 user, 128 in, 128 out |     0.42 -> 0.26 s |    74.8 -> 35.2 ms |               12.9 -> 27.1 |
| 1 user, 2048 / 128      |     5.36 -> 1.46 s |    75.9 -> 36.4 ms |                8.5 -> 21.0 |
| 4 users, 512 / 256      |     4.48 -> 1.78 s |   131.5 -> 58.3 ms |               26.3 -> 61.5 |
| 1 user, 8192 / 64       |     20.8 -> 5.37 s |    76.0 -> 37.0 ms |                 2.5 -> 8.3 |

Raw files: `results/bench/baseline-5101e0e/`, `results/bench/final/`.

Model decode latency outside vLLM, traced, host inputs in and logits out (`scripts/bench_decode.py --trace`):

| Batch |    Base |   Final | Final tok/s total |
| ----: | ------: | ------: | ----------------: |
|     1 | 49.7 ms | 32.9 ms |              30.4 |
|     4 |   82 ms | 54.6 ms |              73.3 |

Breakdown of one batch-1 step: base trace execution 39.3 ms plus 9.1 ms logit readback; final trace execution 31.6 ms plus 0.6 ms readback. Device kernel sum is about 25 ms; the remaining ~6.5 ms are on-device gaps between ~2,200 ops.

Prefill outside vLLM: 2048 tokens 5.36 -> 1.49 s, 8192 tokens 20.8 -> 5.5 s, 30,000 tokens 29 s.

## Correctness

Teacher-forced against the fp32 HF reference. A single run of these metrics moves by 0.04-0.07 KL when the input moves by one BF16 ulp, because a top-8 routing flip propagates through attention. So every comparison below is three runs: seed 0 unperturbed, seeds 1-2 with 2^-9 relative noise on the embeddings.

| Metric | Base KL (3 seeds) | Final KL (3 seeds) | Top-1 base / final |
| --- | --- | --- | --: |
| Decode, 4 segments x 128 steps | 0.332 / 0.420 / 0.419 (mean 0.390) | 0.415 / 0.346 / 0.392 (mean 0.384) | 78.2% / 78.0% |
| Prefill, 2048 tokens | 0.205 / 0.200 / 0.273 (mean 0.226) | 0.251 / 0.194 / 0.238 (mean 0.228) | 85.4% / 85.5% |

Functional checks: `tests/smoke_consistency.py` decode-vs-prefill PCC 0.99997-0.99999 (batch 4 and the 1100-token sliding ring), `scripts/vllm_consistency.py` PASS (4 concurrent greedy chat requests stopping at different lengths equal the same requests alone, twice), sampled requests produce coherent text.

## Where decode time went, and what changed

| Change | Commit | Effect |
| --- | --- | --- |
| Row-major logits | `af22fec` | readback 9.1 -> 0.6 ms per step |
| Fused router (`topk_router_gpt`), slot weights scale `h`, `fast_reduce_nc` | `f32639d` | routing 218 -> 80 us per layer |
| Expert gate/up padded 352 -> 384 per chip | `f32639d` | gate_up sparse matmul on 24 cores instead of 8, 172 -> 118 us per layer |
| Fused QKV rows: one norm, one RoPE, height-sharded K/V straight to the cache | `ea2e6a0`, `1573ed6` | 17 -> 8 ops per layer; fp32 post-processing keeps accuracy |
| Expert-major prefill | `b85f4ab` | each chunk reads one expert once; TTFT 3.6x faster |
| Decode only the active rows | `807f056` | 1 user stopped paying for 4 vLLM rows |
| Top-k sampler for vLLM decode | `99cc25a` | removes a 19 ms full-vocab sort per step |
| DRAM-sharded wo, padded MLP | `c115c7a` | 0.8 ms per step |

## Remaining bottlenecks (final, batch 1, per layer)

- Expert sparse matmuls, 118 + 60 us. `sparse_matmul` loops over the selected experts on every core and splits only N, so it reads about 80 GB/s. Batch 4 needs 32 slots and spends most of its extra 22 ms here.
- Dense matmuls (QKV 66 us, MLP 51 + 31 us, wo 34 us) and the LM head (1.2 ms per step): 150-190 GB/s.
- Two all-reduces, about 110 us, latency-bound (all_gather plus a local sum is no faster).
- `topk_router_gpt`, 69 us, of which 65 us is its top-k and softmax stage.
- 6 RMSNorms with shard conversions, about 100 us; 88 tiles only split 8 ways on the 8x8 grid.
- ~73 ops per layer at ~3 us of device gap each.

100 tok/s (10 ms per token) is out of reach with these kernels: reading every weight once per token is already about 6 ms per chip at full DRAM bandwidth.

## Options tried and not kept

- LM head split into 4 matmuls: 1.22 vs 1.25 ms (the profile's 2.49 ms was an outlier).
- DRAM-sharded QKV: 71 -> 76 us with the conversion.
- `addcmul` in RoPE: correct, but no measurable gain.
- `moe_compute` (DeepSeek fused MoE): needs expert parallelism (replicated mapping), 132 us for batch 1 and 455 us for batch 4 per layer, approximate GELU. A possible batch-4 path; see `scripts/bench_moe_compute.py`.
- BFP8 experts: only both halves help (KL 0.13), and that does not fit.

## Bugs fixed

- Since `9d00363`, inputs of 17-32 tokens (vLLM's prefill warmup) crashed the indexed sparse matmul, so the vLLM server did not start.
- Constants created after a trace capture could be overwritten by that trace's replay; they are now allocated before the first capture.
- `tests/fake_checkpoint.py` seeded with `hash()`, which differs per process.
- Raw completion prompts get no `<bos>` from this tokenizer, and Gemma's output degrades without it. Use the chat endpoint or add `<bos>`.

## tt-scope notes for a separate task

- The device buffer still overflows at the default 1,000 programs for one decode step (2,200 ops per chip); `--programs 6000` was needed, and the check then passed. A hint with the measured op count would help.
- Op-to-op gaps in tracy captures (about 490 ms per step here) are profiler and host dispatch time, not the 6.5 ms of device gaps a traced step shows. A capture mode for trace replay would measure the gaps that matter.
- One capture reported the LM head at 2.49 ms, which is 1.22 ms in isolation. Cross-check outliers before acting on them.
- Raw device logs are about 5 GB per decode capture.

## Reproduce

    pixi run python scripts/bench_decode.py --batch 1 --steps 32 --trace
    pixi run python scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace [--perturb 1]
    pixi run python scripts/noise_floor.py --tokens 2048 final 1 2
    pixi run serve-gemma4   # then: pixi run bash scripts/bench.sh <label>; pixi run python scripts/vllm_consistency.py
