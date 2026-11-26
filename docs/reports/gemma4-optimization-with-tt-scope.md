# 使用 tt-scope 分析的 Gemma 4 推論最佳化

Gemma 4 26B-A4B 在單張 N300d（兩顆 Wormhole chips）上透過 vLLM 提供文字生成服務。模型採 Google QAT checkpoint，主要權重使用 BFP4／BFP8。後期 runtime 為 TTNN 0.82.0、vLLM 0.32.0 與 tt-scope 0.5.0。[Runtime 紀錄][stack] 程式範圍截至 2026-11-26 查核的 [tt-models 版本][head]。

各項結果只屬於自己的配對比較。早期與後期的 runtime、生成設定不同，所以本文不計算跨階段的總加速倍率。[階段紀錄][summary] 術語見 [Glossary](../glossary.md)，tt-scope 操作細節、服務穩定性、最新服務結果與驗證限制見[附錄](gemma4-optimization-with-tt-scope-appendix.md)。

## tt-scope 怎麼找到線索

Agent 用 tt-scope 錄製 Tracy captures，再依 chip 與 layer 內的位置整理 operations。在 Metal Trace replay 下，報表的 `OP TO OP LATENCY` 包含 host 在 replays 之間的時間。把這一欄直接加總，會得到約 395 ms 的 gap，但整個 decode step 只有約 34 ms。Agent 改用 device firmware start/end cycles，算出 replay 內的 gap 約 0.8 ms，因此調查轉向 kernels 本身的成本。[分析紀錄][findings] [分析腳本][breakdown]

Tracy 負責定位 device 成本。Expert weights 的排列與跨 requests 的去重由程式和 kernel tests 檢查，host 路徑則用 `py-spy` 與分段計時。是否採用改動，依未開 profiler 的配對計時或服務測試決定。

## 主要最佳化

### Readback 不傳 padding rows

**觀察**：Round 1 把 batch size 1 的 `model.decode` 單步分段計時。Trace 執行花 39.3 ms，logits readback 另花 9.1 ms。程式檢查發現，原程式以 tile layout 傳回 logits。只有一列是有效輸出，卻讀回 32 rows，其中 31 rows 是 padding。

**做法**：在 `ttnn.to_torch` 前加入 `ttnn.to_layout(..., ttnn.ROW_MAJOR_LAYOUT)`，只傳需要的 rows。[Code diff][rowdiff]

**結果**：當時 batch size 1 的 readback **9.1 → 0.6 ms**。這個數字只包含裝置結果傳回 host 的時間，不是整個 decode step。[實驗紀錄][r1]

### 按 DRAM bank 讀取 expert weights

**觀察**：Batch size 4 的 Tracy capture 顯示，一個 decode step 中 chip 0 的 kernel time 合計約 50.9 ms，其中 `ttnn.sparse_matmul` 占 **20.5 ms**，gap 只有 **0.86 ms**。所以瓶頸在 operations 本身，不在 operations 之間的 gap。原本的 `ttnn.sparse_matmul` 逐 tile 讀取所選的 expert weights。[改動前 capture][rawbatch]

**做法**：改用自訂 kernels，按 DRAM bank 分配讀取工作，以較大封包搬運權重，並把 GELU activation function、乘上 routing weights 與 8 個 experts 的輸出加總，合併進同一組 kernels。[Source code][expertcode]

**結果**：改動後的 Tracy capture 中，chip 0 的 kernel time 合計 **50.86 → 31.45 ms**。[改動後 capture][rawbatchafter] 未開 profiler 的配對計時中，batch size 4 直接呼叫 `model.decode` 的單步時間 **56.49 → 35.42 ms**，包含輸入輸出傳輸，不包含 token selection。[計時紀錄][batched]

### 密集儲存 hidden state，合併相關運算

**觀察**：另一輪 batch size 4 的 Tracy capture 顯示，chip 0 的正規化、加法與格式轉換約占 **7 ms**，60 次 all-gather 另占約 **3.3 ms**。原本每個 request 的 hidden state 有 2,816 個 components，占用 88 個 tiles，但每個 tile 只用其中一列。Batch size 4 時，四個 requests 各用一列，其餘列都是 padding。[改動前 capture][rawboundary]

**做法**：把每個 hidden state 密集裝入 3 個 FP32 tiles，再合併跨 chip 加總、RMSNorm 和 residual 加法。同一改動也把 residual 從 BF16 改為 FP32，所以結果同時反映這個精度改動。[Source code][boundarycode]

**結果**：Chip 0 的 kernel time 合計 **30.55 → 24.83 ms**。[改動後 capture][rawboundaryafter] 未開 profiler 時，batch size 4 直接呼叫 `model.decode` 的單步時間 **34.95 → 29.49 ms**，包含輸入輸出傳輸與 host softcap。這組比較也包含多列 router 的調整，不能全部歸因於密集儲存或合併運算。同日 concurrency 4 的服務 TPOT **36.17 → 30.77 ms**，output throughput **96.9 → 111.3 tokens/s**。[實驗紀錄][boundaries] [比較版本][boundarycompare]

### Requests 共用 expert weights，prompts 批次 prefill

**觀察**：舊程式對每個 request 分別讀取選中的 8 個 experts。紀錄中 batch size 4 的 32 次 expert 選擇（4 × 8）只涉及 29–30 個 distinct experts。這個數字來自 `expert_stream_check.py` 的 kernel 測試：輸入是 `torch.randn` 產生並正規化的 synthetic hidden states，權重是實際模型權重。它不是真實流量，也不是 Tracy 統計，原實驗摘要稱為 random prompts，以腳本為準。[測試腳本][expertcheck] 另一項計時測試檢查逐筆處理 prompts 的成本。[Prefill 紀錄][round3]

**做法**：Router 在每個 decode step 列出不重複的 experts，每個 expert 的權重只讀一次，供相關 requests 共用。Prefill 改成多筆 prompts 批次執行。[Expert code][expertcode] [Prefill code][prefillcode]

**結果**：Batch size 4 直接呼叫 `model.decode` 的時間 **28.81 → 26.01 ms**，包含 host softcap。改動後的版本同時包含 16-row 執行配置與 Q4_0 routing fix，所以這組數字不能全部歸因於共用讀取。支援的 decode batch size 從 4 擴至 16，也需要暫存空間與執行配置的調整。[Shared reads 紀錄][shared] 15 筆各 144-token 的 prompts 從逐筆處理 **3.30 秒 → 批次 1.45 秒**。[Prefill 紀錄][round3]

### Greedy 只讀回最高分候選

**觀察**：原本每步讀回完整 logits，再由 host 執行 argmax。`py-spy` 的 stack samples 中，`decode_forward` 有 **89%** 落在 `LmHead.read`。這段時間包含等待裝置，不全是傳輸。分段計時另確認完整 logits readback 和 host argmax 的成本。[診斷紀錄][round3] [分段測試][reassessment]

**做法**：LM head 的每個 worker 只負責一段 vocabulary 的 logits。改動後，每個 worker 在自己那段 vocabulary 中找出分數最高的 token，host 只讀回這些候選，每個 worker、每個 request 各一個，再從中選出分數最高者。只有所有 active requests 都是 greedy 且未要求 logprobs 時，才走這條路徑。[read_greedy][greedycode] [啟用條件][adaptercode]

**結果**：在同一程式版本交錯測試三種選詞方式：完整 logits 加 PyTorch argmax、完整 logits 加 NumPy argmax、裝置候選加 host 合併。Batch size 16 的 greedy step 分別為 **41.24 ms**、**39.54 ms** 與 **36.01 ms**。這組計時包含 token selection，與前面 `model.decode` 的計時範圍不同。6,048 次逐 request、逐 step 的比較，選出的 token 都一致。[Audit][audit]

## 其他最佳化

數字只適用於各項的受測條件。「`model.decode` 單步」的計時範圍見 [Glossary](../glossary.md#profiling-與計時的補充)。

- **Router fusion**：Round 1 的 profile 中，routing 每 layer 需 218 µs。改用合併的 `topk_router_gpt`，並一併調整 slot weights 的套用與 `fast_reduce_nc` 後，每 layer **218 → 80 µs**（[`f32639d`][f32639d]）。[紀錄][r1]
- **Expert gate/up padding**：Round 1 的 profile 顯示 `sparse_matmul` 只用 8 cores。每顆 chip 的 gate/up 寬度從 352 補到 384，改用 24 cores 後，每 layer **172 → 118 µs**（[`f32639d`][f32639d]）。[紀錄][r1]
- **QKV processing**：原本 query、key、value 各自做正規化與 RoPE。合併成一次正規化、一次 RoPE，K/V 直接寫入 cache 後，每 layer **17 → 8 operations**（[`ea2e6a0`][ea2e6a0]、[`1573ed6`][1573ed6]）。[紀錄][r1]
- **Prefill 按 expert 處理**：原本 prefill 的每個 chunk 重複讀取 expert weights。改成每個 chunk 對每個 expert 只讀一次（[`b85f4ab`][b85f4ab]）。2,048-token TTFT **5.36 → 1.46 秒**是整個 Round 1 的結果，不是此項的獨立收益。[紀錄][r1]
- **只計算 active requests**：原本 vLLM 為單一 request 也計算四個 rows。改成只計算 active rows（[`807f056`][807f056]），紀錄未另列此項的獨立加速值。[紀錄][r1]
- **Top-k sampler**：原本 host sampling 每步排序整個 vocabulary，約 **19 ms**。改成只處理 top-k candidates，移除這次排序（[`99cc25a`][99cc25a]），但整步不等於快 19 ms。[紀錄][r1]
- **All-reduce 改寫**：Microbenchmark 中，`ttnn.all_reduce` 處理 2,816 個值的 padded row 需 **43 µs**。改用 `all_gather_async` 加本地 `fast_reduce_nc` 後為 **34 µs**，batch size 1 的 replay time **30.9 → 30.1 ms**（[`2104d70`][2104d70]）。[紀錄][r2]
- **Dense MLP 按 DRAM bank 讀取**：tt-scope 的 trace breakdown 中，原本的 `ttnn` MLP matmuls 每 layer 約 83 µs 加 16 µs 的銜接運算。改走 expert 用的按 bank 讀取 kernels 後，batch size 1 的 replay time **19.7 → 18.65 ms**（[`350a68c`][350a68c]）。[紀錄][r2]
- **Multi-row router**：`router_check.py` 顯示 `topk_router_gpt` 在任何 batch size 都要 69–72 µs，另有四個 reshape 與 typecast。把自訂 router 延伸到 1–4 rows 後，batch size 4 的 `model.decode` 單步 **34.8 → 33.8 ms**（[`7a3ccaf`][7a3ccaf]）。密集儲存 hidden state 那組比較也包含此改動。[紀錄][multirouter]
- **只對候選做 softcap**：原本 host 對每列 262,144 個 logits 做 softcap，每列 0.34 ms。Softcap 在數學上單調遞增，所以改成先依原始 logits 選出 candidates，只對 candidates 做 softcap。服務 TPOT 在 concurrency 1 為 **18.46 → 18.14 ms**，concurrency 4 為 **37.75 → 36.44 ms**（[`8aa505e`][8aa505e]）。[紀錄][softcap]
- **Runtime update**：同日在 `0d46091` 上比較 TTNN 0.81.1+conda.8 與 0.82.0+conda.1，正確性測試結果相同。換版後 concurrency 1 的服務 TPOT **18.10–18.16 → 17.46–17.47 ms**。這是軟體版本變更，不算自訂 kernel 的收益。[紀錄][stack]
- **Q/K normalization and RoPE**：Round 3 的 Tracy captures 中，`ttnn` 的 q/k 正規化與 RoPE 鏈每 layer 需 68–120 µs。合併成一個 28–47 µs 的 operation，並改用單一 torch thread 建立輸入後，batch size 16 的 `model.decode` 單步 **56.05 → 49.81 ms**，包含這兩項改動（[`87f99ff`][87f99ff]）。[紀錄][round3]
- **Router top-k 分散到多核心**：Round 3 的 Tracy captures 中，batch size 16 的 router operation 需 **145 µs**。改由多個 router cores 分擔 top-k 後為 **69 µs**，batch size 16 的 `model.decode` 單步 **49.80 → 47.56 ms**，輸出不變（[`7c565e5`][7c565e5]）。[紀錄][round3]
- **SDPA chunk size**：原本 decode SDPA 每塊讀 64 個 cache 位置。Page table 涵蓋長度可被 128 整除時改讀 128 個，其餘維持 64。Batch size 1、2K 位置的 `model.decode` 單步 **18.33 → 18.03 ms**（[`0875726`][0875726]）。[紀錄][round3]
- **LM head 按各 chip 配置**：Round 3 的 Tracy captures 顯示，`ttnn` 的 DRAM-sharded matmul 依 chip 0 的 DRAM 配置放置 readers，chip 1 的 LM head 因此較慢。改用依各 chip DRAM bank 配置的 kernels 後，chip 1 的 LM head operation **2.41 → 1.68 ms**，batch size 1 的 `model.decode` 單步 **16.95 → 15.84 ms**（[`41ef565`][41ef565]）。[紀錄][round3]
- **Expert weights 按各 chip 配置**：Round 2 的 profile 顯示，expert 讀取的 readers 也依 chip 0 的 DRAM 配置放置，chip 1 每步慢 0.5 ms。改成各 chip 分別配置後，batch size 4 的 `model.decode` 單步 **23.21 → 22.78 ms**（[`ed12310`][ed12310]）。[紀錄][r2] [結果紀錄][round3]
- **Output projection**：原本 attention 的 output projection 結果要再經一個 pack operation 轉成密集排列。改用 MLP down projection 的按 bank 讀取 kernels，直接寫出密集向量，並移除 pack operation 後，batch size 16 的 `model.decode` 單步 **47.48 → 46.49 ms**（[`1b4b150`][1b4b150]）。[紀錄][round3]
- **RoPE input reuse**：程式檢查發現，每列輸入含 21 個 FP32 tiles，其中有重複的 RoPE tiles。共用相同的 tiles 後降為 11 個，batch size 16 的完整 logits 單步 **37.68 → 37.18 ms**（[`9d2b250`][9d2b250]）。[分析紀錄][reassessment] [Audit][audit]
- **Small all-gather**：3 tiles 與 6 tiles 的小型 all-gather 原本使用兩個 fabric workers，也就是負責 chip 間傳輸的 workers。Microbenchmark 中改用一個 worker 後，分別 **10.94 → 7.97 µs**、**12.03 → 10.28 µs**，batch size 1 的完整 logits 單步 **15.39 → 15.21 ms**（[`e87dbd9`][e87dbd9]）。[分析紀錄][reassessment] [Audit][audit]

[head]: https://github.com/Stanley5249/tt-models/commit/342e751a47665fda8e52bf407cb13f7f544d10a1
[summary]: ../experiments/2026-11-23-results-summary.md
[findings]: ../experiments/2026-11-19-tt-scope-findings.md
[breakdown]: ../../scripts/profile/trace_breakdown.py
[reassessment]: ../experiments/2026-11-25-gemma4-decode-readback.md
[rowdiff]: https://github.com/Stanley5249/tt-models/commit/af22fec026acd72a26439f0d46bd2cf118da8473
[r1]: ../experiments/2026-11-17-decode-round-1/report.md
[r2]: ../experiments/2026-11-19-decode-round-2/report.md
[expertcode]: ../../src/tt_models/models/gemma4/expert_stream.py
[batched]: ../experiments/2026-11-21-batched-expert-streams.md
[rawbatch]: ../experiments/2026-11-21-batched-expert-streams/decode-b4-before.txt
[rawbatchafter]: ../experiments/2026-11-21-batched-expert-streams/decode-b4-after.txt
[boundarycode]: ../../src/tt_models/models/gemma4/boundary.py
[boundaries]: ../experiments/2026-11-21-compact-boundaries-multi-row.md
[boundarycompare]: https://github.com/Stanley5249/tt-models/compare/d565603...333c5e8
[rawboundary]: ../experiments/2026-11-21-compact-boundaries-multi-row/decode-b4-main.txt
[rawboundaryafter]: ../experiments/2026-11-21-compact-boundaries-multi-row/decode-b4-compact.txt
[expertcheck]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/scripts/kernels/expert_stream_check.py#L82-L106
[round3]: ../experiments/2026-11-23-decode-serving-round-3.md
[prefillcode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L332
[shared]: ../experiments/2026-11-23-shared-expert-reads.md
[greedycode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/lm_head.py#L226
[adaptercode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L400
[audit]: ../experiments/2026-11-25-decode-readback-audit.md
[stack]: ../experiments/2026-11-22-shared-stack.md
[multirouter]: ../experiments/2026-11-21-router-op-multi-row.md
[softcap]: ../experiments/2026-11-21-lazy-softcap.md
[f32639d]: https://github.com/Stanley5249/tt-models/commit/f32639d
[ea2e6a0]: https://github.com/Stanley5249/tt-models/commit/ea2e6a0
[1573ed6]: https://github.com/Stanley5249/tt-models/commit/1573ed6
[b85f4ab]: https://github.com/Stanley5249/tt-models/commit/b85f4ab
[807f056]: https://github.com/Stanley5249/tt-models/commit/807f056
[99cc25a]: https://github.com/Stanley5249/tt-models/commit/99cc25a
[2104d70]: https://github.com/Stanley5249/tt-models/commit/2104d70
[350a68c]: https://github.com/Stanley5249/tt-models/commit/350a68c
[7a3ccaf]: https://github.com/Stanley5249/tt-models/commit/7a3ccaf
[8aa505e]: https://github.com/Stanley5249/tt-models/commit/8aa505e
[87f99ff]: https://github.com/Stanley5249/tt-models/commit/87f99ff
[7c565e5]: https://github.com/Stanley5249/tt-models/commit/7c565e5
[0875726]: https://github.com/Stanley5249/tt-models/commit/0875726
[41ef565]: https://github.com/Stanley5249/tt-models/commit/41ef565
[ed12310]: https://github.com/Stanley5249/tt-models/commit/ed12310
[1b4b150]: https://github.com/Stanley5249/tt-models/commit/1b4b150
[9d2b250]: https://github.com/Stanley5249/tt-models/commit/9d2b250
[e87dbd9]: https://github.com/Stanley5249/tt-models/commit/e87dbd9
