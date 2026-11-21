# Q4_0 prefill by dequantizing to BFP (rejected)

**Question.** Q4_0 serving needs a prefill path. Is dequantizing the Q4
experts to a temporary BFP tensor for today's sparse matmul cheap enough?
Gate: 3.5 ms per layer.

**Setup.** `perf/profiling` (`91d4461`). Earlier options were already out:
ttnn's indexed sparse matmul has no per-(block, output) scale,
`ttnn.multiply` with row-broadcast steps took 34 ms per layer for gate|up
alone, the step matrices are not low rank (a rank-8 log fit gives 18 dB
weights), and BFP4 kept for prefill next to Q4 for decode needs 6 GB more
per chip. The new kernel ran one work item per (expert, 32-K group, column
tile), multiplied each integer tile by its step row and packed the result.
Timed by tt-scope on layer 0, both matrices.

**Result.**

| Dequant output | Per layer | Per prefill chunk (30 layers) |
|---|---:|---:|
| BFP4 | 3.7 ms | ~110 ms |
| BFP8 | 5.9 ms | ~180 ms |

The packer truncates when it packs BFP formats, even with
`bfp8_pack_precise`: 57% of values matched host round-to-nearest, so the
BFP4 output was less accurate than the host-converted weights. A reader that
reserved 32 tiles in a ring of 24-tile pushes hung the card three times
before it was found.

**Decision.** Rejected: 40-75% more TTFT on a 128-token prompt, and less
accurate. A Q4 prefill needs an expert-major matmul that dequantizes inside
and keeps each weight tile for all of an expert's chunks.

**Data.** None; the kernel was not kept.
