# Expert-major prefill in one-tile chunks

**Question.** Where does a 2,048-token prefill spend its time, and how much
of the expert matmul work is padding?

**Setup.** `7a1dfcc`. tt-scope capture of one 2,048-token prefill of random
tokens, before and after:

    pixi run tt-scope tracy scripts/profile/workload.py --name prefill2048 --programs 12000 -- --prompt-len 2048 --decode-steps 0

**Result.** Before: 1,390 ms kernel time on chip 0. Sparse matmuls 462 ms
(33%), all-gathers 161 ms, expert row gathers 153 ms, slices 81 ms, GELU
69 ms, untilize 65 ms, concat 60 ms; the glue around the experts cost about
as much as the matmuls. With 128-row chunks, 224 chunks held 16,384 real
pairs, so 43% of the matmul rows were padding. Matmul time follows the rows:
1.28 s with 32-row chunks, 1.34 s with 64, 1.54 s with 128.

Changes: chunks are always one tile; each sparse matmul call gathers its own
rows instead of slicing one big gather; padding rows gather token 0 with
weight 0, which drops the appended zero row and its concat. After: 1,110 ms,
sparse matmuls 368 ms, gathers 80 ms, slices 30 ms. Logits unchanged to PCC
1.00000, prefill KL 0.264 (earlier runs 0.20-0.33).

**Decision.** Kept. The chunk count rounds to 16 (`expert_plan.CHUNK_GRANULE`)
because each new count compiles new programs. The all-gathers (162 ms, about
6 GB/s over one Ethernet link) and the gate|up matmul on 24 of 64 cores are
what remain.

**Data.** `2026-11-21-one-tile-expert-chunks/`: `prefill-2048-{before,after}.txt`
(per-op tables), `prefill2048-onetile.json` (accuracy).
