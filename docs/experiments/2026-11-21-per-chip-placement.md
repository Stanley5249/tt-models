# Per-chip stream placement (reverted)

**Question.** Chip 1 runs the DRAM-streaming ops slower than chip 0 (LM head 152 against 117 us, expert down 29.8 against 26.0 us). Is that placement, and does fixing it shorten the step?

**Setup.** `perf/profiling` (`efbe7cf`). tt-scope captures of a traced batch-1 step; the stream ops built with each chip's own core placement through a `MeshProgramDescriptor`.

**Result.** The chips are harvested differently: 8 of 12 DRAM banks have a different nearest worker core. Swapping the placements moved the gap (26.0 / 29.5 us with chip 0's placement, 42.6 / 25.8 with chip 1's), which pins it on placement. Per-chip placement took chip 1's expert down from 29.8 to 25.5 us, but the batch-1 step span only fell 16.30 -> 16.18 ms (0.7%), because the merged router and MLP op got slower on chip 1. Gathering only the real rows before the all-gathers (untilize, row-major gather, tilize) saved 12-17 us on the 5,632-wide gather and nothing on the 2,816-wide one, and the compact boundaries made it moot.

**Decision.** Reverted, too little for the code. The LM head, a ttnn matmul, shows the same chip gap and is out of reach without replacing it.

**Data.** None beyond the per-op tables in [compact boundaries](2026-11-21-compact-boundaries-multi-row/).
