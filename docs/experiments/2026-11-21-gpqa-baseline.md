# GPQA Diamond baseline

**Question.** Is there a task-level quality number for this precision, next
to upstream's QB2 release check?

**Setup.** `feat/serving` (`8893874`). `scripts/accuracy/gpqa_eval.py`,
upstream's QB2 settings and scoring, against a running `serve-gemma4`:

    uv run --no-project --with-requirements scripts/accuracy/gpqa_eval.requirements.txt \
        python scripts/accuracy/gpqa_eval.py --count 40 --out artifacts/gpqa/<label>

**Result.** 26/40 correct, 7/10 on upstream's subset. Upstream gates its
higher-precision QB2 build on 8/10. With 40 sampled answers the 95% interval
is about +-15 points.

**Decision.** A local baseline only, not evidence of equal or worse quality
than upstream. Comparing precision choices needs more questions or seeds.

**Data.** `2026-11-21-gpqa-baseline/gpqa-feat-serving/`.
