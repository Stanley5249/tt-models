# Decode accuracy of the batch-1 and batch-4 paths

`scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <k> --batch <B>`: every row decodes the same teacher-forced README segments; row 0 is scored against the cached fp32 reference. Batch 1 runs the fused compact-FP32 path, batch 4 the batched streams with generic attention and norms.

| Files         | Path    | Mean KL | Mean top-1 |
| ------------- | ------- | ------: | ---------: |
| `b1-k{1,2,3}` | batch 1 |   0.457 |      0.758 |
| `b4-k{1,2,3}` | batch 4 |   0.423 |      0.772 |

The two paths are equally accurate within the noise floor (one perturbed run moves 0.04-0.07 KL), and each is about 0.44 KL from fp32. At a flat position they can still disagree with each other by a similar amount (a 3,500-token code prompt: KL 0.50 between the batch-1 and batch-4 logits), which is why a greedy request can continue differently when it shares decode steps with others.
