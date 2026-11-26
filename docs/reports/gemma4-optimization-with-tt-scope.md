# 使用 tt-scope 分析的 Gemma 4 推論最佳化

vLLM 在單張 N300d 上執行 Gemma 4 26B-A4B，提供文字生成服務。N300d 有兩顆 Wormhole chips。模型採 Google QAT checkpoint，expert weights 使用 BFP4，其他權重使用 BFP8。目前的 runtime 為 TTNN 0.82.0、vLLM 0.32.0 與 tt-scope 0.5.0。[本文描述的程式版本][head]

各項結果只比較該項的改動前與改動後，一組比較同時包含其他改動時，附錄會列出那些改動。各實驗的 runtime 與生成設定不同，所以本文不計算總加速倍率。[結果摘要][summary] 術語與 `model.decode` 的計時範圍見 [Glossary](../glossary.md#profiling-與計時的補充)。各項的實驗日期、commits、其他量測與同一組比較中的其他改動，以及 tt-scope 操作細節、服務穩定性與驗證限制，見[附錄](gemma4-optimization-with-tt-scope-appendix.md)。

## 主要最佳化

### Logits readback 不傳 tile 中的 padding rows

**觀察**：Batch size 1 的 decode step 分段計時中，trace 執行花 39.3 ms，logits readback 另花 9.1 ms。Code review 發現，原程式以 tile layout 把 logits 從 device 讀回 host。讀回的 32 rows 中只有 1 row 是有效輸出，其餘 31 rows 是 padding。[實驗紀錄][r1]

**做法**：Decode 的 readback 路徑先用 `ttnn.to_layout(..., ttnn.ROW_MAJOR_LAYOUT)` 把 device 上的 logits 改為 row-major，再用 `ttnn.to_torch` 只把有效輸出的 rows 從 device 讀回 host。[Code diff][rowdiff]

**結果**：Batch size 1 的 logits readback 耗時為 **9.1 → 0.6 ms**。這個計時只涵蓋 device 結果傳回 host，不是整個 decode step。[實驗紀錄][r1]

### 按 DRAM bank 讀取 expert weights

**觀察**：Batch size 2–4 的 decode 用 `ttnn.sparse_matmul` 逐 tile 讀取 router 選中的 expert weights。Batch size 4 的 Tracy capture 中，`ttnn.sparse_matmul` 讀取 expert weights 的速度只有約 85 GB/s。[實驗紀錄][batched]

**做法**：改用自訂 kernels，按 DRAM bank 分配讀取工作，以較大的區塊讀取 expert weights。GELU、乘上 routing weights 與 8 個 experts 輸出的加總也 fuse 進同一組 kernels。這組 kernels 先用於 batch size 1，再延伸到 batch size 2–4。[Source code][expertcode]

**結果**：Batch size 4 以 `model.decode` 計時的 decode latency 為 **56.49 → 35.42 ms**，expert weights 的讀取速度提高到約 204–226 GB/s。這組 decode latency 也包含 dense MLP 改用同一組 kernels 的效果。[實驗紀錄][batched]

### Hidden state 改放進 3 個 tiles，fuse layers 之間的運算

**觀察**：Hidden size 為 2,816，每個 request 的 hidden state 分散在 88 個 tiles。Batch size 4 時，每個 tile 的 32 rows 只用 4 rows，其餘 28 rows 是 padding。Batch size 4 的 Tracy capture 中，chip 0 在 matmuls 之間的 RMSNorm、加法、layout 轉換、slice 與 concat 約占 **7 ms**。[實驗紀錄][boundaries]

**做法**：Decode 把每個 request 的 hidden state 依序存入 3 個 FP32 tiles。兩顆 chips 以 3-tile 的 all-gather 交換 partial sums，再由各 chip 的 tail op 一次完成本地加總、RMSNorm、residual 加法與 layer scaling。[Source code][boundarycode]

**結果**：Batch size 4 以 `model.decode` 計時的 decode latency 為 **34.95 → 29.49 ms**，concurrency 4 的服務 TPOT 為 **36.17 → 30.77 ms**。[實驗紀錄][boundaries]

### 同一 decode step 中，每個 expert 的 weights 只讀一次

**觀察**：舊程式對每個 request 分別讀取該 request 選中的 8 個 experts，不同 requests 選到同一個 expert 時，同一份 weights 會讀取多次。以 synthetic hidden states 執行 `expert_stream_check.py` 時，batch size 16 的 128 次 expert 選擇只涉及 84–86 個 distinct experts。[測試腳本][expertcheck]Expert 的暫存空間也讓 decode 最多只能處理 4 個 requests。[實驗紀錄][shared]

**做法**：Router 在每個 decode step 列出 distinct experts，每個 expert 的 weights 只讀一次，供選中該 expert 的 requests 共用。同一改動也擴大暫存空間並調整工作分配，讓 decode 最多可處理 16 個 requests。[Source code][expertcode]

**結果**：Batch size 4 以 `model.decode` 計時的 decode latency 為 **28.81 → 26.01 ms**。Server 同時處理的 requests 上限由 4 提高到 16，concurrency 16 的 output throughput 為 **115.7 → 216.2 tokens/s**。[實驗紀錄][shared]

### 多個 requests 的 prefill 放進同一個 batch

**觀察**：vLLM scheduler 在同一個 scheduler step 排入多個新 requests 時，adapter 逐一對每個 request 做 prefill。每次 prefill 都重讀所有 weights，所以短 prefill 每次約有 110 ms 的固定耗時。[實驗紀錄][shared]

**做法**：Adapter 把同一個 scheduler step 中較短的 prefill 片段分組，每組只執行一次模型，projections、MLP 與 experts 一起處理組內所有 rows。Attention 仍把每個 request 的 K/V 寫入該 request 的 KV cache，並只對該 request 的序列計算 attention。[Source code][prefillcode]

**結果**：單獨量測 prefill 時，15 個各 144 tokens 的 prompts 耗時為 **3.30 → 1.45 秒**，concurrency 16 的 TTFT 為 **2.92 → 1.39 秒**。[實驗紀錄][round3]

## 其他最佳化

- **Router fusion**：Profile 中，routing 每 layer 需 218 µs。改用 `topk_router_gpt`，把 router 的 top-k 與對選中 logits 的 softmax fuse 成一個 operation。MoE 把各 expert 的 routing weight 乘到該 expert 的 down projection 輸入上，並以 `fast_reduce_nc` 取代 `ttnn.sum` 加總 8 個 experts 的輸出，routing 每 layer 降到 **80 µs**。[紀錄][r1]
- **Expert gate/up padding**：Profile 顯示 `sparse_matmul` 只用 8 cores 計算 gate/up。每顆 chip 的 gate/up 寬度從 352 補到 384，讓 `sparse_matmul` 可用 24 cores，每 layer **172 → 118 µs**。[紀錄][r1]
- **QKV processing**：原本 decode 分別對 Q、K、V 做 RMSNorm，再只對 Q 與 K 做 RoPE。改為把 fused QKV 輸出當成同一組 rows，用一次 RMSNorm 與一次 RoPE 處理 Q、K、V，V 的 rows 使用 cos 1、sin 0，所以 RoPE 不改變 V 的數值。拆開 heads 後，K 與 V 直接寫入 KV cache，每 layer 的 operations 為 **17 → 8**。[紀錄][r1]
- **Prefill 先按 expert 整理 tokens**：原本 prefill 以每 32 個連續 tokens 為一組，計算這組選到的所有 experts。每組幾乎涵蓋全部 128 個 experts，2,048-token prompt 每組每 layer 約讀 180 MB expert weights。改成 host 從 device 讀回每個 token 的 top-8 expert 選擇，按 expert 把 tokens 整理成 chunks，每個 expert chunk 只讀一次該 expert 的 weights。vLLM 外單獨量測的 2,048-token prefill 為 **5.36 → 1.49 秒**。[紀錄][r1]
- **只計算 active requests**：vLLM 把 decode 輸入補到 `max_num_seqs` rows。`max_num_seqs` 為 4 時，即使只有一個 active request，model 也以 batch size 4 計算。改成 model 從 vLLM 傳入的 decode batch 移除 position 為 -1 的 rows，並為每種 active request 數各錄一個 Metal Trace。只有一個 active request 時，batch size 從 4 降到 1，trace 執行時間為 **62 → 32 ms**。[紀錄][r1]
- **Top-k sampler**：Gemma 預設的 top-k 64、top-p 0.95 下，vLLM 的 CPU sampler 每個 decode step 在單一 PyTorch thread 上排序整個 vocabulary，約需 19 ms。改成由 adapter 從傳回的 logits 取出 top-k tokens，再套用 vLLM 的 temperature、top-k 與 top-p 規則。Concurrency 1 的服務 TPOT 為 **52.8 → 36.4 ms**。[紀錄][r1]
- **All-reduce 改寫**：Microbenchmark 中，`ttnn.all_reduce` 處理 2,816 個值的 padded row 需 43 µs。改用 `all_gather_async`，再由每顆 chip 用 `fast_reduce_nc` 在本地加總兩顆 chips 的 partial sums，batch size 1 的 replay time 為 **30.9 → 30.1 ms**。[紀錄][r2]
- **Dense MLP 按 DRAM bank 讀取**：tt-scope 的 trace breakdown 中，原本的 `ttnn` MLP matmuls 每 layer 約 83 µs，另有 16 µs 的銜接 operations。Dense MLP 改走 expert 用的按 bank 讀取 kernels 後，batch size 1 的 replay time 為 **19.7 → 18.65 ms**。[紀錄][r2]
- **自訂 router**：Profile 中，batch size 1 的 `topk_router_gpt` 每 layer 約 71 µs，多數時間花在 top-k。改用自訂 router operation，由 11 個 cores 各算一部分 router logits，reducer 加總後在 RISC 上選出 top 8 並做 softmax。自訂 router 先用於 batch size 1，再延伸到 batch size 2–4。batch size 4 以 `model.decode` 計時的 decode latency 為 **34.8 → 33.8 ms**。[Batch size 1 紀錄][r2] [Batch size 2–4 紀錄][multirouter]
- **只對選出的 tokens 做 softcap**：原本 host 對每個 request 的 262,144 個 logits 做 softcap，每個 request 需 0.34 ms。Softcap 在數學上單調遞增，所以 greedy decoding 與 top-k sampling 改用原始 logits 選出 tokens，只對選出的 k 個 logits 做 softcap。Concurrency 4 的服務 TPOT 為 **37.75 → 36.44 ms**。[紀錄][softcap]
- **Q/K RMSNorm、RoPE 與 host 輸入**：Tracy captures 中，`ttnn` 的 Q/K RMSNorm 與 RoPE operations 每 layer 需 68–120 µs。另外，host 在等待 device 後喚醒 PyTorch thread pool，每個 decode step 花 1–2 ms。改成 decode 把 Q/K 的 RMSNorm 與 RoPE fuse 成一個 operation，host 另用單一 PyTorch thread 建立 `model.decode` 的輸入。兩項改動合計，batch size 16 以 `model.decode` 計時的 decode latency 為 **56.05 → 49.81 ms**。[紀錄][round3]
- **Router top-k 分散到多核心**：Tracy captures 中，batch size 16 的 router operation 需 145 µs。改由多個 router cores 分擔 top-k，輸出不變，batch size 16 以 `model.decode` 計時的 decode latency 為 **49.80 → 47.56 ms**。[紀錄][round3]
- **SDPA K chunk size**：原本 decode SDPA 每次從 KV cache 讀 64 個 positions。改為在 page table entries 數乘上 KV block size 可被 128 整除時，每次讀 128 個 positions，否則仍讀 64 個。Batch size 1、2,048 positions 時，以 `model.decode` 計時的 decode latency 為 **18.33 → 18.03 ms**。[紀錄][round3]
- **各 chip 依自己的 DRAM bank 位置選擇讀取 cores**：`ttnn` 的 DRAM-sharded matmul 與原本的 expert kernels 都依 chip 0 的 DRAM bank 位置選擇讀取權重的 cores。兩顆 chips 的 harvesting 不同，所以 chip 1 讀得較慢，all-gather 又讓兩顆 chips 都等待較慢的 chip 1。Tracy captures 中，chip 1 的 LM head 每個 decode step 比 chip 0 慢 0.55 ms。LM head 改用依各 chip 配置的讀取 kernels 後，batch size 1 以 `model.decode` 計時的 decode latency 為 **16.95 → 15.84 ms**。[配置調查][r2] [改動後量測][round3]
- **Attention output projection 直接輸出 3-tile hidden state**：原本 output projection 的結果要再經一個 pack operation，轉成 hidden state 的 3-tile 排列，batch size 16 時 pack operation 需 36 µs。Output projection 改用 MLP down projection 的按 bank 讀取 kernels，直接寫成 3-tile 排列，並移除 pack operation，batch size 16 以 `model.decode` 計時的 decode latency 為 **47.48 → 46.49 ms**。[紀錄][round3]
- **Greedy decoding 時，device 只傳回每段 vocabulary 的最高 logit**：原本 `LmHead.read` 每個 decode step 都把完整 logits 從 device 讀回 host，再由 host 執行 argmax，batch size 16 讀回完整 logits 需 3.683 ms。改成 LM head 的每個 worker 先在 device 上找出每個 request 在自己那段 vocabulary 的最高 logit 與 token ID，只把這些值傳回 host，host 再從中選出分數最高的 token。只有所有 active requests 都使用 greedy decoding 且未要求 logprobs 時，adapter 才走這條路徑，checkpoint 預設的 sampling 設定下沒有收益。Batch size 16 含 token selection 的 decode latency 為 **41.24 → 36.01 ms**，兩種做法選出的 token 都一致。[分段計時][reassessment] [Audit][audit]
- **RoPE input reuse**：Code review 發現，每個 request 的 host 輸入含 21 個 FP32 tiles，其中有內容相同的 RoPE tiles。每個 request 的輸入改為相同的 RoPE tiles 只存一份後，降為 11 個 tiles。Batch size 16 只計到完整 logits、不含 argmax 的 decode latency 為 **37.68 → 37.18 ms**。[分析紀錄][reassessment] [Audit][audit]
- **Small all-gather**：3 tiles 與 6 tiles 的小型 all-gather 原本每條 link 使用兩個 fabric workers。改為小型 all-gather 每條 link 只用一個 fabric worker，更大的資料仍用兩個。Batch size 1 只計到完整 logits、不含 argmax 的 decode latency 為 **15.39 → 15.21 ms**。[分析紀錄][reassessment] [Audit][audit]

[head]: https://github.com/Stanley5249/tt-models/commit/342e751a47665fda8e52bf407cb13f7f544d10a1
[summary]: ../experiments/2026-11-23-results-summary.md
[reassessment]: ../experiments/2026-11-25-gemma4-decode-readback.md
[rowdiff]: https://github.com/Stanley5249/tt-models/commit/af22fec026acd72a26439f0d46bd2cf118da8473
[r1]: ../experiments/2026-11-17-decode-round-1/report.md
[r2]: ../experiments/2026-11-19-decode-round-2/report.md
[expertcode]: ../../src/tt_models/models/gemma4/expert_stream.py
[batched]: ../experiments/2026-11-21-batched-expert-streams.md
[boundarycode]: ../../src/tt_models/models/gemma4/boundary.py
[boundaries]: ../experiments/2026-11-21-compact-boundaries-multi-row.md
[round3]: ../experiments/2026-11-23-decode-serving-round-3.md
[prefillcode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L332
[shared]: ../experiments/2026-11-23-shared-expert-reads.md
[audit]: ../experiments/2026-11-25-decode-readback-audit.md
[multirouter]: ../experiments/2026-11-21-router-op-multi-row.md
[softcap]: ../experiments/2026-11-21-lazy-softcap.md
[expertcheck]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/scripts/kernels/expert_stream_check.py#L82-L106
