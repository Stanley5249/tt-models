# Trace safety: buffers allocated after a capture

**Question.** Why did identical raw-token requests to the server disagree from their first token, while a standalone repeat was bitwise stable?

**Setup.** Prefill and decode inputs dumped from the server and replayed standalone in the server's call order, on `2a525b3`. Fixes in `b82754d` and `26f078d`, each also verified on the old code. `scripts/profile/memory_report.py` for the leak test.

**Result.** A buffer allocated after a decode trace capture can overlap the trace's intermediates, and every replay overwrites it. Two existed:

- **Decode inputs and lazy constants.** Each batch size's inputs were allocated just before its own capture, and some constants (rotate_half's matrix, MoE masks) on first use. Unchanged page tables are not re-copied, so a replay could read corrupted block ids, which hung the card.
- **`ttnn.all_reduce` in prefill.** It caches 12 KB of L1 semaphores per new input shape. The first prefill of a new length after capture cached them in trace memory, and the next prefill of that length returned garbage (logits off by 28).

**Decision.** `_prepare_traces` allocates every batch size's inputs and runs each once before the first capture. Prefill uses decode's reduction: `all_gather_async` on semaphores allocated at load plus a local sum. Rule for new code: allocate everything a trace touches before the first capture. The prefill reduction moved decode KL by 0.036 at 3 seeds, which [matched noise](2026-11-21-matched-noise.md) traced to rounding.

**Data.** None; the repro scripts were one-off.
