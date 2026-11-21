# Weight formats for the N300

**Question.** Which device format per weight group fits 24 GB on two
Wormhole chips and loses the least accuracy?

**Setup.** `e25e6f7` (first model), `d91e1cd` (expert gate|up and down
emulated separately). `scripts/accuracy/precision_study.py --tokens 512`:
the HF reference with chosen weight groups rounded to BFP formats on the
host, teacher-forced on README text, scored against fp32. One run each.

**Result.** All 128 experts take 12.8 GB at BFP4 and 24.3 GB at BFP8, so BFP8
experts do not fit. KL against fp32 at 512 tokens:

| Config | KL | Top-1 |
|---|---:|---:|
| HF BF16 compute, exact weights | 0.163 | 91.4% |
| BFP8 everywhere | 0.131 | 90.2% |
| BFP4 experts, BFP8 dense and LM head | 0.349 | 84.7% |

BFP8 experts helped only when both gate|up and down were BFP8 (KL 0.13).
LoFi and HiFi2 fidelity on the BFP4 experts differed within the noise.

**Decision.** BFP4 experts with LoFi, BFP8 attention, MLP, LM head and KV
cache (`config.Precision` defaults). The QAT checkpoint is Q4_0, which BFP4
cannot hold; see [BFP4 and mixed precision](2026-11-21-bfp4-mixed-precision.md)
and [Q4_0 decode experts](2026-11-21-q4-decode-experts.md).

**Data.** `2026-11-17-weight-precision/`: `precision_study.json`,
`precision_study_emu2.json` (split expert emulation), and the first device
comparison `compare-L30-p128-d16-initial.json`.
