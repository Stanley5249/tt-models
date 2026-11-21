# Q4_0 decode experts (experimental)

**Question.** If decode reads the checkpoint's own Q4_0 integers and steps instead of BFP4 roundings, how much accuracy comes back, at what cost?

**Setup.** `ab185fe`. TT-Metal has no Q4_0 or int4 tile format on Wormhole (`MxInt4` and `MxFp4` unpack only on Quasar), so the kernels (`src/tt_models/models/gemma4/expert_q4.py`, `kernels/expert_q4_*`) keep the integers in BFP4 tiles, which hold -7..7 exactly, and the steps as BF16 tiles. The rare -8 (0.2% of values, 6% of blocks) is clamped to -7. Device accuracy by `scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <k> [--precision expert_q4=1]`, 3 seeds each; kernel checks and times by `scripts/kernels/q4_stream_check.py` with tt-scope captures; host emulation by `scripts/accuracy/q4_emulation.py`.

**Result.** Device decode against fp32:

| Experts                | KL per seed           | Mean KL | Mean top-1 |
| ---------------------- | --------------------- | ------: | ---------: |
| BFP4 (served)          | 0.438 / 0.479 / 0.454 |   0.457 |      0.758 |
| Q4_0, -8 clamped to -7 | 0.302 / 0.215 / 0.341 |   0.286 |      0.813 |

Host emulation agrees on the ranking:

| Experts                             | fp32 compute, 2,048 tokens | BF16 compute, 512 tokens |
| ----------------------------------- | -------------------------: | -----------------------: |
| `q4exact` (q * s)                   |                      0.025 |                    0.174 |
| `q4shift` (+1 offset where it fits) |                      0.086 |                        - |
| `q4clamp` (the device kernels)      |                      0.110 |                    0.250 |
| `bfp4` (served)                     |                      0.197 |                    0.347 |

Kernels, layer 0, one token: gate|up output SNR 16-19 dB (BFP4) against 28-34 dB (Q4), down 20-25 against 35-40 dB; device time 47.8 -> 60.7 us and 27.7 -> 41.2 us, about 0.8 ms per token at batch 1 (4%). The Q4 outputs match host math on the clamped integers to 52 dB, so the remaining gap is the clamp. tt-scope drove three kernel changes: doubling local copies of one zero tile (82 -> 6 us of zero-fill), reading steps one slot at a time, and keeping the stock 11-step reader. A +1 offset per block would store 95% of the -8 blocks exactly but recovered only 0.024 of the 0.085 clamp loss.

**Decision.** Kept as an opt-in accuracy mode (`Precision(expert_q4=True)`, batch-1 decode only), not served: prefill has no Q4 path ([prefill dequantization](2026-11-21-q4-prefill-dequant.md)) and two copies of the experts do not fit in DRAM.

**Data.** `2026-11-21-q4-decode-experts/`: `decode-{bfp4,q4}-k{1,2,3}.json`, `emulation-fp32-2048.json`, `emulation-bf16-512.json`.
