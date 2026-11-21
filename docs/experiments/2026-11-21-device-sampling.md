# Device sampling, async decode and MTP (not built)

**Question.** The plugin's async decode overlaps host work with the next step if the model samples on the device. Is it worth building? Is MTP speculative decoding?

**Setup.** `feat/quality-sampling`, reassessed in `feat/quality-upstream` (`f3b345b`). A one-off split of a traced step (30 steps, median), `ttnn.topk` microbenchmarks, and a device top-k candidate path.

**Result.** Host work per step that async decode could hide:

| Part                                          |      Batch 1 |    Batch 4 |
| --------------------------------------------- | -----------: | ---------: |
| Host inputs (embedding, RMSNorm, RoPE tables) |      0.46 ms |    0.72 ms |
| Input copies                                  | 0.67-0.80 ms |    0.71 ms |
| Trace execution                               |      15.7 ms |    31.0 ms |
| Logit readback and softcap                    | 1.19-1.40 ms | 2.5-2.7 ms |

Sampling on the device eats most of it. `ttnn.topk` has a cliff: k = 32 over 32,768 columns takes 0.29 ms but over 65,536 16.7 ms, and a chip holds 131,072 logits. `ttnn.sampling` takes BF16 logits (greedy near-ties would differ from the fp32 argmax) and at most 32 users. The token feedback needs the embedding on the device (0.74 GB per chip out of the KV pool). Net about 0.6-1 ms per step at batch 1.

The cheapest exact path, BF16 per-chip logits viewed as 32 x 4,096 rows, top-64 per row, and a read of the candidates only (`device-topk.txt`): 0.51 against 0.73 ms at batch 1, 2.43 against 2.53 ms at batch 4, before the fp32 gather a greedy-exact version needs.

MTP: this checkpoint has no MTP weights; upstream's drafter is a separate 4-layer model. Verifying K drafts needs a multi-token decode or a prefill starting off the 128-token block grid, which the prefix cache forbids. A batch-4 step costs 2x a batch-1 step, so the best case is about 15-20%.

**Decision.** Not built. The win needs a custom LM-head kernel that keeps per-core top-k candidates, plus a device embedding: a TT-Metal-sized change. The lazy softcap took the cheap part ([lazy softcap](2026-11-21-lazy-softcap.md)).

**Data.** `2026-11-21-device-sampling/device-topk.txt`.
