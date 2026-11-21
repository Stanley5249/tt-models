# Accuracy noise floor

**Question.** How much does decode KL against fp32 move between runs that
should be equivalent, so that a change can be judged?

**Setup.** `29f6b5a`. `scripts/accuracy/compare.py --decode-steps 128
--offsets 0 1024 2048 3072 --trace --perturb <seed>`: 4 README segments, a
128-token prompt and 128 teacher-forced decode steps each. Seed 0 is plain;
seeds 1-2 add 2^-9 relative noise to the embeddings, about one BF16 ulp.
Prefill used the same perturbation over 2,048 tokens (`noise_floor.py`, last
at `7f72451`).

**Result.** One run moves 0.04-0.07 KL. A top-8 routing flip at a near-tie
propagates through attention, so ulp-level changes reorder later tokens.
Perturbed runs of the same code differ as much as two different commits did.

**Decision.** Every accuracy comparison uses at least 3 seeds and reports
the mean. A difference under about 0.05 KL with 3 seeds is not evidence.
[Matched noise](2026-11-21-matched-noise.md) later keyed the noise to the
tokens so that runs are comparable across commits.

**Data.** `2026-11-17-accuracy-noise-floor/`: decode runs at `ea2e6a0`
(fp32 post-processing, seeds 0-2), the fused-QKV variants and
`prefill2048-noise.jsonl`.
