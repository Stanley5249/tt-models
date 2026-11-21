# Softcap only the sampled candidates

**Question.** Capping all 262,144 logits per row on the host costs time every
step. Can the cap be skipped for most of them?

**Setup.** `ebea5de`. The final softcap `30 tanh(x / 30)` keeps the order,
so greedy and top-k sampling can pick candidates from the raw logits and cap
only those. Serving TPOT by `scripts/serve/bench_tis.sh`.

**Result.** The cap took 0.34 ms per row (1.35 ms at batch 4). Serving TPOT
18.46 -> 18.14 ms at 1 user, 37.75 -> 36.44 ms at 4.

**Decision.** Kept. The decode step returns uncapped logits plus the cap
function; logprobs and vLLM's own host sampler (penalties, min-p, larger
top-k) still get the whole capped row.

**Data.** None beyond the commit.
