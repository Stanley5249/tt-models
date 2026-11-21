# Every KV layer in vLLM: prefix caching, chunked prefill, preemption

**Question.** Can vLLM own all 30 layers' KV cache, so that prefix caching, chunked prefill and preemption work without adapter state?

**Setup.** `c712a83` (hybrid KV), `0ecdc9e` (block-grid starts), `a485d83` (alignment check), `d2772a5` (preemption check). Before, vLLM paged only the 5 global layers and the adapter kept a 1,024-token ring per batch slot for the 25 sliding layers. Checked with the random-weight device tests (`tests/test_device.py`), `scripts/serve/serve_check.py`, and `serve_check.py preempt` on a server started with `GEMMA4_KV_POOL_GIB=0.2 ... --max-model-len 4096`.

**Result.**

- `get_kv_cache_spec` declares sliding layers as `SlidingWindowSpec` and global layers as `FullAttentionSpec`. vLLM forms 5 sliding groups and 1 global group with their own page tables. A block id belongs to one group, so the i-th layer of every group shares one K and one V buffer; to keep pages equal, global layers get 128-token blocks next to 64-token sliding blocks, and each layer reads the shared buffer through its own geometry.
- A prefill chunk start must lie on the block grid (128 tokens with `--block-size 64`). vLLM schedules on it, so shared prefix blocks are never rewritten. The chunk start goes to the device as a tensor, so a new offset builds no programs (2.3 s before on a cold JIT cache).
- Chunked prefill and a prefix hit match one full prefill at PCC >= 0.99998. A cached 6,144-token prefix gives 323 ms TTFT for 256 new tokens.
- Preemption: four requests outgrow a 313-block pool, vLLM preempts one and resumes it by recomputation, and all four equal their solo runs over 1,536 tokens (twice). Code prompts were dropped from the check because their flat distributions diverge from batch numerics alone.

**Decision.** Kept. The model holds no per-request state, so `release_request` and `slot_remap` are gone. The KV pool is a fixed 2.5 GiB per chip (`GEMMA4_KV_POOL_GIB`).

**Data.** Serving runs in [serving backend](2026-11-21-serving-backend.md).
