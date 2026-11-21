# Router op for 2-4 decode rows

**Question.** `topk_router_gpt` took 69-72 us per layer at any batch plus four reshape and typecast ops. Can batch 1's router op take 1-4 rows?

**Setup.** `08c1278`. `scripts/kernels/router_check.py --rows 1 2 3 4` against `topk_router_gpt` and an fp32 reference.

**Result.** Each core writes its B partial rows into rows `slot B + b` of the reducer tile, a 0/1 sum matrix (`J[b, r] = 1` when `r mod B = b mod B`) adds each token's rows, the reducer RISC picks every row's top 8, and the op writes the streams' flat row directly. At 3-4 rows, 8 cores of 11 K tiles replace 11 of 8, so the partial rows fit one tile; a version that kept 11 cores over two reducer tiles was no faster. Cost 37-42 us at 1-2 rows, 54-68 us at 3-4. Same expert sets as `topk_router_gpt` in 94-100% of rows; where they differ `topk_router_gpt` is the one off (max weight error against fp32: 0.0007 for the op, 0.11 for `topk_router_gpt`). Batch-4 step 34.8 -> 33.8 ms, measured before the compact boundaries.

**Decision.** Kept, merged with the MLP gate|up as at batch 1.

**Data.** None beyond the commit.
