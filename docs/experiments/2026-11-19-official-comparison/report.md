# Gemma 4 26B-A4B on N300d vs the official Tenstorrent results

Subject: this repo at `a527dad` (model code identical to `f158df1`), ttnn 0.81.1+conda.8, vLLM 0.31 with vllm-tt-plugin 0.1.0, one N300d (two Wormhole chips, 1x2 mesh), CPU governor `performance`. Official sources were read on 2026-11-19: tt-metal `upstream/main` at `bbba07b2874` and tt-inference-server `main` at `0ad0f4be90`.

Labels used below: **[M]** measured here, **[P]** published by Tenstorrent, **[E]** analytical estimate, **[H]** hypothesis.

## 1. Executive summary

- **How fast.** Under the same client that grades the official release targets (tt-inference-server's `vllm bench serve` construction: chat endpoint, random 128-token prompts, 128 output tokens, ignore-EOS, greedy), our server gives **55.0 tok/s/user** (mean TPOT 18.17 ms, three repetitions 54.9-55.3) and 237 ms TTFT for one user **[M]**. Four users get 17.3 tok/s/user and 63.2 tok/s total **[M]**.
- **Against the official numbers.** The closest official point is the release gate for the canonical code on Blackhole QB2 (four chips): 45.69 tok/s/user at 128/128, one user, same client **[P]**. Our single-user decode is 1.20x that, on half the chips of an older generation. But newer official QB2 configurations already reach 53.5 (dedicated QB2 code, plain decode) and 62.7 tok/s/user (canonical code with MTP speculation) **[P, config comments, not gates]**. TTFT is 3-5x worse than QB2 at every input length, and the official builds serve 32-128 users against our 4.
- **What is supported.** On one N300d, our implementation reaches a single-user decode speed in the same range as Tenstorrent's 4-chip Blackhole QB2 builds. The two differences we can quantify are that it reads about **half the weight bytes per token** (BFP4 experts, BFP8 dense weights) and that, at TPOT, it uses about 31% of nominal DRAM bandwidth against about 14% for the QB2 gate **[E]**. How much each one contributes cannot be separated.
- **What cannot be concluded.** We cannot say our code is faster or more efficient than upstream. The hardware, precision, checkpoint, serving features and code revisions all differ, and none of them can be held equal on N300 (the official precision does not fit in 24 GB). Our precision is also measurably less exact than the official one, and we have no task-level quality result.

## 2. Performance comparison

### Systems

| System | Chips | DRAM | Nominal DRAM BW | Code | Weights (decode) |
| --- | --- | --: | --: | --- | --- |
| Ours, N300d | 2 Wormhole | 24 GB | 576 GB/s | `gemma4_tt`, TP=2 | QAT ckpt; experts BFP4, dense + LM head BFP8, residual FP32 |
| QB2 (P300x2) | 4 Blackhole | 128 GB | 2048 GB/s | canonical `models/demos/gemma4`, TP=4 | base ckpt; experts BFP8, attention/MLP/LM head BF16 |
| Galaxy | 32 Blackhole | - | - | canonical, TP=8 x EP=4, 4 lanes x 32 slots | as above |
| T3K | 8 Wormhole | 96 GB | 2304 GB/s | canonical, 1x8, composite (unfused) path | BF16 everywhere (no 1x8 override in `precision_overrides.json`) |

Precision comes from `precision_overrides.json` and `create_tt_model(dtype=bfloat16)` at `upstream/main` **[P, code]**; the T3K CI run may have used an older revision.

### Single-user decode, 128/128

| Source | System | tok/s/user | TTFT ms | Boundary | Comparability |
| --- | --- | --: | --: | --- | --- |
| **Ours [M]** | N300d | **55.0** (54.9-55.3) | 237 | vLLM, TIS client, greedy | - |
| Ours, sampled (temp 1, top-k 64) [M] | N300d | 53.8 (53.6-53.9) | 252 | same, server-default sampling | - |
| QB2 gate, 2026-10-28 [P] | QB2 | 45.69 | 71.9 | same client; async scheduling, device sampling, no MTP | **Partially comparable** |
| Gemma 4 README [P] | QB2 | 46.2 | 144 | "local serving on this branch", not gated | Partially comparable |
| TIS spec comment, 2026-11-04 [P] | QB2 | 53.5 | - | dedicated `gemma4_26b_a4b_qb2` v0.26, greedy c1 | Partially comparable |
| TIS spec comment, 2026-11-04 [P] | QB2 | 62.7 | - | canonical + MTP K=3 (now the release default) | Not comparable (speculation) |
| README release run [P] | Galaxy | 22.1 | 450 | vLLM, 128-slot lane engine | Not comparable |
| TIS gates, 2026-10-26 / 10-31 [P] | Galaxy | 37.04 / 38.26 | 103 / 139 | plain lanes / MTP lanes | Not comparable |
| README Metal demo CI [P] | T3K | 19.9 | 398 | `text_demo.py` host-synced step loop, batch 1 | See below |
| Ours, standalone [M] | N300d | 57.0 (17.53 ms) | - | `bench_decode.py --trace`, host in, logits out | Partially comparable with T3K |
| Ours, device only [M] | N300d | (15.55 ms) | - | blocking trace execution | Device time only |

**Why "partially comparable" for the QB2 gate.** The workload and the timing boundary match: TIS's client sends random tokens through the chat template (~144 input tokens, our runs averaged 144.3), forces greedy, and grades `1000 / mean TPOT`. Still different: our client is vLLM 0.31 (the gate used 0.13, so we pass `--temperature 0` explicitly), chip count and generation, 2x weight bytes on QB2, a different checkpoint, async scheduling and device sampling on QB2 (neither on ours), and code a month older than upstream `main`.

**T3K.** Same chip generation, and our standalone loop (host inputs in, logits out, synced per step) is the closest boundary to the Metal demo's per-step timing; the demo also samples on host. Ours is 2.9x faster on a quarter of the chips. But T3K reads about 2.3x our bytes per token **[E]**, runs Wormhole without any of the canonical fused programs, and has 8-way TP communication, so this ranks configurations, not implementations.

### Longer prompts and concurrency [M, 3 repetitions each]

| Point | Ours tok/s/user | Ours TTFT ms | QB2 gate tok/s/user | QB2 TTFT ms |
| --- | --: | --: | --: | --: |
| 2048/128, c1 | 51.8 (51.6-51.9) | 1573 | 44.59 | 310.5 |
| 8192/128, c1 | 50.8 (50.4-51.1) | 6953 | 44.41 | 1298.6 |
| 128/128, c4 (ours) / c32 (QB2) | 17.3, **63.2 total** | 736 | 31.67, 835 total | 891 |

Decode slows by 7.7% from 128 to 8K tokens on ours and 2.8% on QB2. TTFT is 5.1-5.4x slower than QB2 for long prompts (3.3x at 128): our prefill is eager, unchunked and runs on two Wormhole chips. The c4 row is not comparable to c32. It does show that our batch-4 decode (56.6 ms standalone **[M]**) still runs the pre-optimization path, so four users barely beat one in total throughput. 32K/128 is not runnable on our 32K-token server.

Variability: TPOT spread across repetitions is under 1% at every point; p99 ITL stays within 1.5 ms of the mean at c1 **[M]**. The QB2 gates carry a 0.2 tolerance and come from one run each **[P]**.

## 3. Performance analysis

### Where our 18.2 ms goes [M]

From the tt-scope `tracy` capture of the current code (`results/final/trace-breakdown-final.txt`, chip 0, kernel time per step), re-checked today with a blocking trace time of 15.55 ms:

| Part | ms/step | Share |
| --- | --: | --: |
| Weight-streaming ops: QKV 2.11, wo 1.24, router + MLP gate\|up 1.52, MLP down 0.62, experts 2.19 | 7.67 | 42% |
| Attention core: q/k norm + RoPE, cache updates, SDPA, head ops | 2.50 | 14% |
| Layer tails (norms, residual, scalars) + pack | 1.93 | 11% |
| Cross-chip all-gathers (2 per layer) | 0.85 | 5% |
| LM head + final ops | 2.06 | 11% |
| Gaps and waiting for chip 1 (trace 15.55 minus chip-0 kernel 15.01) | 0.54 | 3% |
| Host and vLLM (TPOT 18.17 minus trace 15.55) | 2.62 | 14% |

Chip 1 takes 0.5 ms more than chip 0 per step, and the all-gathers make both chips wait for it. Gaps between ops under trace are 0.34 ms per step.

### Bandwidth [E]

`scripts/weight_traffic.py` multiplies active parameters (3.82B per token: experts 1.43B, attention 1.11B, MLP 0.54B, LM head 0.74B) by bytes per weight:

| Config | GB per token | Floor at nominal BW | Measured step | Effective BW |
| --- | --: | --: | --: | --: |
| Ours | 3.36 | 5.8 ms | 15.55 ms device / 18.17 ms TPOT | 215 / 179 GB/s (37% / 31%) |
| QB2 canonical | 6.31 | 3.1 ms | 21.9 ms TPOT | 288 GB/s (14%) |
| T3K canonical (BF16) | 7.64 | 3.3 ms | 50.3 ms per step | 152 GB/s (7%) |

Our weight-streaming ops alone run at about 172 GB/s per chip, 60% of nominal. The 2x byte difference is the largest single factor between us and QB2, but it does not explain the result by itself: QB2 has 3.6x our nominal bandwidth and is far from its floor. **[H]** At batch 1 the QB2 step is bound by per-op latency, four-way collectives and host work rather than DRAM, which is the regime where our fused, bank-local ops and two-chip collectives help. These are estimates from nominal bandwidth; nothing here measured DRAM traffic on either system.

**Why the precision gap cannot be removed here [E].** All 128 experts take 12.8 GB at BFP4 and 24.3 GB at BFP8, and N300 has 24 GB in total. The official precision does not fit, which matches the upstream support matrix (26B-A4B on N300: not supported, still true on `main`).

### Accuracy

- Ours vs the fp32 HF reference of the same QAT checkpoint, teacher-forced decode, 6 seeds: top-1 78.3%, KL 0.368 **[M, from round 2]**.
- Weight-only emulation over 512 tokens (`results/accuracy/precision_study*.json`, one run, noise ~0.05 KL) **[M]**: BFP4 experts + BFP8 dense: top-1 84.7%, KL 0.349. BFP8 everywhere: 90.2%, KL 0.131. HF BF16 compute: 91.4%, KL 0.163.
- **[H]** The official QB2 precision (BFP8 experts, BF16 dense) should sit near or above the BFP8 row, so its logits are likely 2-3x closer to the reference than ours. The QAT checkpoint targets q4_0 (32-value blocks with a scale), not BFP4 (16-value shared exponent), so it does not guarantee BFP4 robustness.
- Functional checks today **[M]**: decode vs prefill PCC 0.99998; four concurrent greedy requests match the same requests alone, twice (`scripts/vllm_consistency.py` PASS).
- Official task-level evidence exists (Galaxy 26B-A4B GPQA Diamond 79.8; dedicated QB2 GPQA 9/10) **[P]**. We have none, so we cannot claim equal quality.

## 4. Implementation differences

Our history starts at `3b4f247`, an independent TTNN implementation that borrowed specific choices from `models/demos/gemma4` and tt_transformers (cited in comments). It is not a fork of the official tree, which on our fork's `stable` branch was last updated 2026-10-20 (`18b718f`). Serving TPOT went 132 ms (first vLLM version, eager) -> 74.7 (`5101e0e`) -> 35.2 (`5887694`) -> 18.2 ms now **[M]**. Since that `stable` snapshot, upstream added 73k lines to `models/demos/gemma4`, including the whole `custom_ops/` tree (19k lines of fused programs, mostly Blackhole-only or off on Wormhole).

| Technique (ours) | Measured effect here | Upstream equivalent | Transferable |
| --- | --- | --- | --- |
| **Expert streams**: per-DRAM-bank reads of the 8 routed BFP4 experts, GELU and routing weights fused, slot sum in the down's K loop | trace 30.1 -> 24.1 ms; experts 75 -> ~190 GB/s | gpt-oss `fused_decode` streams (Blackhole only, our cited source); canonical Gemma 4 uses sparse_matmul + `geglu_decode` + `moe_combine` | **[H]** Yes for Wormhole: the canonical Wormhole builds run no fused programs, and our kernels use `matmul_tiles` with 32x32 tiles, which works without Blackhole's `custom_mm` |
| **Compact boundaries**: FP32 residual as 3 dense tiles, 3-tile all-gather, one tail op per boundary | 24.1 -> 19.7 ms; gather 34 -> 10-14 us | canonical `fused_norm_decode` (5 programs per layer, Blackhole); gpt-oss `DecodeBoundary` (flat vector, fused into the next matmul) | Batch-1 only. The packing helps where a row is a tiny share of a tile; upstream's gpt-oss version goes further |
| **Router op** with in-op logits, top-8 and softmax on RISC, merged with the MLP gate\|up on disjoint cores | 71 -> 36 us; merge 16.5 -> 15.66 ms | canonical `router_topk` (ranking after a matmul) | The disjoint-core merge of a latency-bound and a bandwidth-bound op is the one idea we did not find upstream |
| Fused q/k norm + RoPE | 18.65 -> 17.7 ms | canonical `qkv_norm_rope` | Already upstream |
| DRAM-sharded LM head (16 splits) and wo | 31.5 -> 30.9 ms | tt-metal #50648 | Already upstream |
| All-reduce as `all_gather_async` + local sum | 30.9 -> 30.1 ms | async persistent CCL in canonical | Already upstream |
| Softcap and sampling on host | - | device softcap and sampler, async scheduling | Upstream is ahead |

Missing on our side: prefix caching, chunked prefill, async decode, prefill trace, MTP speculation, more than 4 users, context beyond 32K, multimodal. Our streams cover batch 1 only.

## 5. Conclusions

- **Strongest defensible claim.** On one N300d, with BFP4 experts and BFP8 dense weights, our implementation serves Gemma 4 26B-A4B at 55.0 tok/s/user (128/128, one user, greedy) under the client that grades the official targets. That is above the official QB2 canonical gate (45.69) and close to the dedicated QB2 build (53.5), with half the chips and roughly half the weight bytes per token.
- **Limitations.** Different hardware, precision, checkpoint and serving features; lower numerical fidelity than the official precision and no task quality result; 3-5x slower TTFT; weak multi-user scaling; the newest official default (MTP, 62.7) is faster than ours.
- **Upstream relevance.** The Wormhole expert-stream kernels and the disjoint-core router merge are worth offering for the canonical Wormhole (T3K) path, where fused programs are off. Everything else already exists upstream in a Blackhole form. Any proposal needs validation beyond batch 1 and beyond N300.
- **Most useful follow-up.** Run the official QB2 GPQA subset (`gemma4_26b_a4b_qb2/tests/benchmark.py`, 10 questions, threshold 0.8) against our server. That is the one missing piece before calling our precision an equivalent serving configuration.

## Evidence

- Serving: `results/compare/tis-match/` (vLLM bench JSON with per-request detail; `warmup` passes discarded), made by `scripts/bench_tis.sh`. Server log `logs/serve-compare.1316.log` (untracked). The server used the working tree's `pixi.toml`, which adds the gemma4 tool and reasoning parsers to `serve-gemma4`.
- Standalone: `results/compare/standalone/` (`trace_time.txt`, `decode-b1.json`, `decode-b4.json`).
- Official sources: `results/compare/official/` (TIS performance reference for 26B-A4B, the QB2 spec in force at the 2026-10-28 gate (TIS `4f529a6c`) and the current MTP-default spec). The QB2 gate came from tt-shield run 40120766326 at tt-metal `b6e3bd3e4a8`; the README rows cite tt-shield runs 39894754540 (Galaxy) and 39925769041 (T3K CI).
- Estimates: `scripts/weight_traffic.py`, output in `results/compare/weight-traffic.txt`.
- Profiles and accuracy: `results/final/`, `results/accuracy/`, `docs/decode-optimization-report.md`.
