# Q4_0 for the dense weights (deferred)

**Question.** Every 2-D tensor in the checkpoint is Q4_0, including attention, dense MLP, router and the tied embedding. Their BFP8 copies are what the decode streams and LM head read most. Would Q4_0 help them?

**Setup.** `43d6d89`. BF16-compute emulation with BFP4 experts, 512 tokens:

    EMU_DENSE=<mode> EMU_DTYPE=bfloat16 EMU_TOKENS=512 pixi run python scripts/accuracy/q4_emulation.py <out.json> bfp4

**Result.**

| Dense weights                | KL vs fp32 |
| ---------------------------- | ---------: |
| BFP8 (served)                |      0.391 |
| Q4_0, -8 clamped             |      0.561 |
| Q4_0, +1 shift where it fits |      0.348 |
| Q4_0 exact                   |      0.311 |

Q4_0 would cut those weights' traffic by about 40%, roughly 2 ms of a 15.7 ms batch-1 step.

**Decision.** Deferred. The clamp that is enough for the experts is worse than BFP8 here, so the shift correction (an extra activation-sum matmul per block group) is needed, plus new stream kernels for QKV, o_proj and the 131,072-column LM head, plus BFP8 copies kept for prefill.

**Data.** `emulation-bf16-512.json` in [Q4_0 decode experts](2026-11-21-q4-decode-experts/).
