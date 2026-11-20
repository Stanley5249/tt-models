# Decode accuracy with matched noise (feat/serving)

`scripts/compare.py --decode-steps 128 --offsets 0 1024 2048 3072 --trace --perturb <seed>`,
teacher-forced against the cached fp32 reference, with the embedding noise keyed
by each call's tokens (before `scripts/compare.py` drew it from one generator, so
the trace warm-up's extra embed call shifted every later draw; the files in
`../stateful-noise/` used that scheme and are not comparable across commits).

| Files | Code | Mean KL | Mean top-1 |
|---|---|---:|---:|
| `k{1,2,3}-r2` | `2a525b3` (round-2 model, current deps) | 0.421 | 0.777 |
| `k{1,2,3}-head` | `1d61c2b` | 0.457 | 0.758 |

Seed 2 per commit (`b2-*`): 0.426 at `b82754d`, 0.479 from `26f078d` on, and
0.428 at `1d61c2b` with prefill reverted to `ttnn.all_reduce`
(`b2-head-ttnnar`, bitwise the round-2 result). The whole difference comes from
the prefill reduction: gather + local sum differs from `ttnn.all_reduce` in 0.5%
of values and stays within the bf16 rounding bound (max error 0.0156 against
0.0176). Prefill-only KL at 512 tokens moved the other way (0.384 -> 0.319, seed
0). Per-segment decode KL moves in both directions, as ulp-level changes do
(see the noise-floor notes); a small real effect cannot be ruled out with 3 seeds.
