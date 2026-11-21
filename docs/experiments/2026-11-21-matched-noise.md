# Decode accuracy with matched noise

**Question.** Decode KL against fp32 rose from 0.421 (round-2 model) to
0.457 after the serving work. Is that a regression?

**Setup.** `327b29c` keys `compare.py`'s embedding noise to each call's
tokens. Before, it drew from one generator, so the trace warm-up's extra
embed call shifted every later draw and runs were not comparable across
commits. `scripts/accuracy/compare.py --decode-steps 128 --offsets 0 1024 2048
3072 --trace --perturb <seed>`, seeds 1-3, and a per-commit bisect at seed 2.

**Result.**

| Code | Mean KL | Mean top-1 |
|---|---:|---:|
| `2a525b3` (round-2 model, current deps) | 0.421 | 0.777 |
| `1d61c2b` | 0.457 | 0.758 |

Seed 2 per commit: 0.426 at `b82754d`, 0.479 from `26f078d` on, and 0.428
with prefill reverted to `ttnn.all_reduce`, bitwise the round-2 result. The
whole difference comes from the prefill reduction
([trace safety](2026-11-21-trace-safety.md)): the gather plus local sum
differs from `ttnn.all_reduce` in 0.5% of values and stays within the BF16
rounding bound (max error 0.0156 against 0.0176). Prefill-only KL at 512
tokens moved the other way (0.384 -> 0.319, seed 0).

**Decision.** Not a regression: an ulp-level reordering of the same math,
which the [noise floor](2026-11-17-accuracy-noise-floor.md) predicts. 3 seeds
cannot exclude a small effect. Accuracy runs from here on use matched noise.

**Data.** `2026-11-21-matched-noise/`: `matched-noise/` (comparable runs,
with its own README) and `stateful-noise/` (the old scheme, not comparable
across commits).
