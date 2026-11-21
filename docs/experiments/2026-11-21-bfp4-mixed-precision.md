# BFP4 cannot hold the QAT grid; mixed precision not kept

**Question.** The checkpoint is quantization-aware. Can a BFP4 layout or rounding recover its grid, or can a few layers afford BFP8 experts?

**Setup.** `feat/quality-sampling` (`16b7a3b`). Weight SNR of layer 3's experts on the host; CPU emulation of the fp32 reference with bit-exact torch BFP4 experts (`scripts/accuracy/q4_emulation.py`, 2,048 README tokens, and 512 tokens).

**Result.** Every 32-input block of the expert, MLP and attention weights is an integer multiple of one learned step (integers -8..7, 6% of the blocks use -8), stored in BF16. BFP4 shares a power-of-two exponent per 16 outputs with 15 levels, so it cannot land on a learned step's 16 levels:

| Format                                   |   Weight SNR |
| ---------------------------------------- | -----------: |
| BFP4 as stored (blocks along outputs)    |      20.2 dB |
| BFP4 with the best exponent per block    |      20.4 dB |
| BFP4 with blocks along inputs            | 20.4-20.8 dB |
| Per-output-channel rescaling before BFP4 | 18.9-20.0 dB |
| BFP8                                     |      43.2 dB |

BFP4 experts cost KL 0.197 against fp32 at 2,048 tokens (0.341 at 512, where BF16 compute alone is 0.163), the largest single loss.

Mixed precision: the KV pool has room for about five layers' BFP8 experts (0.2 GB per chip each, about +8% TPOT for five). Exact experts in layers 0-5 moved emulated KL 0.197 -> 0.189, in layers 2, 7, ..., 27 to 0.218; one exact layer moved it by -0.02 to +0.08. All inside the noise.

**Decision.** No BFP4 variant and no layer choice is supported by the data, so experts stay BFP4 at LoFi. Exact weights need kernels that read Q4_0 blocks: [Q4_0 decode experts](2026-11-21-q4-decode-experts.md).

**Data.** Emulation scores in [Q4_0 decode experts](2026-11-21-q4-decode-experts/).
