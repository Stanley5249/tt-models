# Prefill length buckets

**Question.** A prompt length the server had not seen built new programs on its first prefill. How much did that cost, and which padding avoids it?

**Setup.** `d07601a` (128-token buckets warmed at startup), `28d3a0b` (whole tiles for short first pieces). `scripts/serve/ttft_lengths.py` sends one-token chat requests at new lengths and times the whole request.

**Result.** On a fresh `feat/serving` server, eight chat prompts at new lengths took 3.4-4.9 s on average, single requests up to 9.7 s; the same lengths took 1.39 s once built. With buckets: 1.17-1.37 s. 256-row buckets were tried first and cost a 144-token prompt 56 ms of TTFT (238 -> 294 ms), hence whole tiles for first pieces up to 512 tokens.

Two shapes still vary inside a bucket. The KV fill builds per 32-row length, which measured as nothing (0.46, 0.38, 0.39 s for 300, 333 and 365 tokens). The expert chunk count depends on the routing, so a new count builds a program once (about 0.4 s); padding to the worst case would cost every request compute.

Standalone prefill of README text over the milestone (one call, min of 2): 512 tokens 0.514 -> 0.450 s, 2,048 1.581 -> 1.269 s, 8,192 5.618 -> 5.101 s.

**Decision.** Kept: `config.prefill_rows` pads to whole tiles for a first piece up to 512 tokens and to multiples of 128 otherwise, and `warmup_model_prefill` builds every length up to the scheduler's chunk size (36 s at 2,048).

**Data.** `2026-11-21-prefill-length-buckets/quality-sampling/`: serving runs of `feat/quality-sampling` and `ttft-unseen-lengths.txt`.
