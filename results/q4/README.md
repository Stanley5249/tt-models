# Q4_0 expert weights (`feat/quality-upstream`)

The QAT checkpoint's routed experts are integers q (-8..7) times a learned BF16
step per 32-input block. The device stored them as BFP4, which cannot hold
that grid. These files compare BFP4 with the checkpoint's own Q4_0 values.

## Device decode accuracy

`scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace
--perturb <k> [--precision expert_q4=1]`: 4 README segments, a 128-token
prompt, 128 teacher-forced decode steps each, scored against the fp32
reference (`scripts/acc_summary.py`). The Q4 runs also prefill through the Q4
streams, one token at a time.

| Files | Experts | KL per seed | Mean KL | Mean top-1 |
|---|---|---|---:|---:|
| `decode-bfp4-k{1,2,3}` | BFP4 (current) | 0.438 / 0.479 / 0.454 | 0.457 | 0.758 |
| `decode-q4-k{1,2,3}` | Q4_0, -8 clamped to -7 | 0.302 / 0.215 / 0.341 | 0.286 | 0.813 |

One perturbed run moves KL by 0.04-0.07, so the gap (0.17) is real.

## Host emulation (no device)

`scripts/q4_emulation.py`: the HF reference with every expert weight replaced,
scored on README text against the fp32 reference.

| Experts | fp32 compute, 2,048 tokens | BF16 compute, 512 tokens |
|---|---:|---:|
| checkpoint (`none`) | 0 (it is the reference) | 0.153 |
| `q4exact` (q * s) | 0.025 | 0.174 |
| `q4shift` (+1 offset where it fits, else clamp) | 0.086 | - |
| `q4clamp` (the device kernels) | 0.110 | 0.250 |
| `bfp4` (the device today) | 0.197 | 0.347 |

`emulation-fp32-2048.json` and `emulation-bf16-512.json` hold the scores; the
fp32 BFP4 row is the earlier `feat/quality-sampling` run. With fp32 compute any
perturbation of the near-exact weights flips near-ties, so that column
exaggerates small weight errors; the BF16 column is closer to the device.

## Kernel numbers (layer 0, one decode token)

`scripts/bench_q4_stream.py`, SNR of the outputs against float64 host math
with the checkpoint weights; device time from a tt-scope capture
(`pixi run tt-scope tracy --name q4both --programs 3000 scripts/bench_q4_stream.py -- --seeds 1`).

| Op | BFP4 SNR | Q4 SNR | BFP4 time | Q4 time |
|---|---:|---:|---:|---:|
| gate\|up (act) | 16-19 dB | 28-34 dB | 47.7 / 47.9 us | 59.7 / 61.7 us |
| down (per chip) | 20-25 dB | 35-40 dB | 26.2 / 29.2 us | 41.2 / 41.2 us |

Times are chip 0 / chip 1. The Q4 outputs match host math on the clamped
integers to 52 dB, so the remaining gap is the -8 clamp.

## Prefill dequantization (`perf/profiling`, not kept)

A device kernel that multiplied each integer tile by its step row and packed a
temporary BFP4 or BFP8 tensor for the prefill sparse matmuls, timed by
tt-scope on layer 0 (both matrices): 3.7 ms per layer to BFP4, 5.9 ms to BFP8,
so 110-180 ms more per prefill chunk. The packer truncates to BFP formats
(57% of values matched host round-to-nearest even with `bfp8_pack_precise`),
so the BFP4 output was less accurate than the host-converted weights. Rejected;
see docs/serving.md, Q4_0 decode experts.

## Dense weights (`perf/profiling`)

`EMU_DENSE=<mode> EMU_DTYPE=bfloat16 EMU_TOKENS=512 scripts/q4_emulation.py ... bfp4`
also replaces the attention, dense MLP and LM-head weights (experts BFP4):
BFP8 0.391, Q4_0 clamped 0.561, +1 shift 0.348, exact 0.311
(`emulation-bf16-512.json`). Every dense and embedding block factors as Q4_0.
