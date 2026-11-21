# Accuracy of the batch-1 and batch-4 decode paths

**Question.** Batch 1 and batches 2-4 ran different kernels. Is the batched path less accurate, and why can a greedy request change when others join?

**Setup.** `ef42b5c`. `scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <k> --batch <B>`: every row decodes the same teacher-forced text; row 0 is scored against fp32. 3 seeds per path.

**Result.**

| Path                                                   | Mean KL | Mean top-1 |
| ------------------------------------------------------ | ------: | ---------: |
| batch 1 (fused compact-FP32 path)                      |   0.457 |      0.758 |
| batch 4 (batched streams, generic attention and norms) |   0.423 |      0.772 |

Equal within the noise, and each is about 0.44 KL from fp32. At a flat position the two can still disagree with each other by as much: KL 0.50 between the batch-1 and batch-4 logits of a 3,500-token code prompt.

**Decision.** No accuracy reason to prefer one path. Batch invariance is not promised (vLLM does not promise it on other backends either), so a greedy request can continue differently at a near-tie when it shares decode steps. `serve_check.py`'s isolation check allows that divergence (`c188deb`).

**Data.** `2026-11-21-batch-path-accuracy/batch-path/` (with its README).
