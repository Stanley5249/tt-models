# Glossary

本頁是 tt-models 新文件中可重用術語的單一維護來源，不依賴 tt-home。報告首次使用術語時可提供簡短說明，完整定義與計時範圍集中在本頁。歷史實驗保留當時用語，引用時依本頁說明其含義。

數值與格式的例子以 Gemma 4 的 N300d 實驗為限，不代表所有模型或 device。模型設定與量測脈絡見 [結果摘要](experiments/2026-11-23-results-summary.md)與[模型設定](../src/tt_models/models/gemma4/config.py)。

| Term | 本文定義 |
| --- | --- |
| Checkpoint | 儲存的模型權重與設定。 |
| Runtime | 執行模型所用的軟體環境。 |
| CPU governor | 控制 CPU 時脈策略的系統設定，performance 是其中一種策略。 |
| Buffer | 儲存資料的記憶體區域。 |
| Constant | 預先建立並重用的固定資料。 |
| Tracy capture | 用 Tracy profiler 記錄 host 與 device 的執行事件。 |
| Metal Trace capture | 把一段 device 指令錄進 trace buffer，供之後 replay。 |
| Exp underflow | 計算指數函數時，結果太小而在所用格式中成為零。 |
| Device | 執行 TTNN operations 的加速卡端；本實驗使用 N300d 的兩顆 Wormhole chips。 |
| Host | 執行 Python、vLLM 與 TTNN 呼叫的 CPU 端。 |
| N300d | 本實驗的 Tenstorrent 雙 chip 加速卡，兩顆 chips 編號為 chip 0 與 chip 1。 |
| Wormhole | N300d 所用的 Tenstorrent chip 架構。 |
| tt-scope | 本實驗使用、整合 device profiling 與報表的工具。 |
| DRAM | device 上存放權重與 KV cache 等資料的記憶體。 |
| TTNN | device 運算函式庫。 |
| TT-Metal | 底層執行與 kernel 程式設計層。 |
| vLLM | 模型服務框架。 |
| Scheduler | 決定下一次模型執行處理哪些 requests 的元件。 |
| Scheduler step | vLLM scheduler 選出 requests 並交給模型執行一次的循環。 |
| Request | 一次有獨立 prompt 與輸出的生成呼叫。 |
| Concurrency | 同時未完成的 requests 數。 |
| Token | 模型文字單位。 |
| Prompt | 輸入內容。 |
| Vocabulary | 模型可輸出的全部 tokens，Gemma 4 有 262,144 個。Logits 對 vocabulary 中的每個 token 各有一個分數。 |
| Decode step | 模型執行一次，為納入的每個 request 計算下一 token 的分數。 |
| Active request | 納入目前 decode step 的 request。 |
| Batch size | 一個 decode step 或 prefill 納入的 requests 數。 |
| Prefill | 處理輸入 prompt。 |
| Chunk | 一次處理的一段資料。Prefill chunk 是一段 prompt tokens，expert chunk 是同一 expert 一次處理的一組 tokens，SDPA K chunk 是 decode SDPA 一次從 KV cache 讀取的一組 positions。 |
| Batched prefill | 把多個 requests 的 prefill 輸入合成一次模型執行，各 request 分別寫入自己的 KV cache。 |
| Sliding-window attention | 每個位置只對最近一段視窗內的 K/V 做 attention 的 layer。 |
| Prefix caching | 重用相同輸入前綴的 KV cache。 |
| Tensor | 多維數值陣列。 |
| Activation | 執行中產生的中間 tensor。 |
| Hidden state | 一個 token 在模型某處的向量表示，本例的 hidden size 為 2,816。 |
| Tile | 本例 32×32 的資料區塊。 |
| Row | Tile 或 tensor 中的一列資料。Decode 時每個 active request 占一個 row，描述 requests 數時改寫 batch size。 |
| Padding | 為滿足儲存形狀補入的資料。 |
| Row-major | 各 row 連續儲存的排列。 |
| Expert | 模型選用的子網路。 |
| Router | 選擇 experts 的運算。 |
| MLP | 多層前饋網路。 |
| Gate/up | MLP 或 expert 中兩個並列的線性轉換，程式中合稱 `gate_up`。Gate 的結果經 GELU 後，與 up 的結果逐項相乘。 |
| Down projection | 把 gate 與 up 的逐項乘積轉回 hidden state 大小的線性轉換。 |
| Routing weight | Router 為某個 token 選中的 expert 算出的輸出權重。 |
| GELU | 作用在 gate projection 結果上的 activation function。 |
| DRAM bank | device 記憶體分區。 |
| Core | 計算核心。 |
| Kernel | device 程式。 |
| Worker | 由特定 operation 分配工作的 core。 |
| Program | TT-Metal 一次啟動的一組 kernels 與設定。 |
| Semaphore | cores 之間同步用的計數器。 |
| DRAM-sharded matmul | TTNN 的 matmul，權重依 DRAM bank 分片，2026-11-23 的量測中，它依 chip 0 的 DRAM bank 位置配置兩顆 chips 讀取權重的 cores。 |
| Fabric worker | 執行 chip 之間傳輸的 worker，例如 all-gather 的傳輸端。 |
| Tail op | 本專案在 layers 之間 fuse 的 operation，包含跨 chip 加總、RMSNorm 與 residual 加法。 |
| Fuse | 把多個計算合成一個 device operation，減少中間資料的搬運與 operation 啟動次數。 |
| Logits | 下一 token 在 vocabulary 上的原始分數。 |
| Readback | 把 device 上的結果傳回 host。 |
| LM head | 將 hidden state 轉成 logits 的輸出層。 |
| Greedy | 選最高分 token。 |
| Argmax | 找最大值位置。 |
| Sampling | 按機率抽樣。 |
| Top-k | 依分數選出最高的 k 個項目，例如 expert IDs 或 token IDs。 |
| Seed | 控制隨機過程的設定值。 |
| Token selection | 從 logits 決定下一個 token 的步驟，包含 argmax 或 sampling。 |
| Logprobs | token 機率的對數。 |
| Softcap | 壓縮 logits 幅度。數學上單調遞增，但 FP32 飽和時可能讓不同分數成為同分。 |
| Residual | 保留並加入新輸出的 hidden state。 |
| RMSNorm | 按 root mean square 調整尺度的正規化。 |
| QKV | attention 的 query、key、value。 |
| RoPE | 旋轉式位置編碼。 |
| SDPA | scaled dot-product attention。 |
| Output projection | attention 輸出的線性轉換。 |
| All-gather | 每顆 chip 收集其他 chips 的資料。 |
| All-reduce | 結合各 chip 的資料，例如加總，再把結果交給各 chip。 |
| BF16 | 16-bit brain floating point。 |
| FP32 | 32-bit IEEE floating point。 |
| BFP4 | Tenstorrent 的 4-bit block floating-point 儲存格式。 |
| BFP8 | Tenstorrent 的 8-bit block floating-point 儲存格式。 |
| Q4_0 | 每組 32 個權重搭配一個 scale 的 4-bit 格式；本專案保留實驗性計算路徑。 |
| QAT | Quantization-aware training，訓練時納入量化影響。 |
| FP32 reference | 以 FP32 計算的對照模型。 |
| KL | KL divergence 的簡寫，比較兩個輸出機率分布的差異，越低越接近 FP32 reference。 |
| Tracy | 時間軸量測工具。 |
| Profiler | 記錄執行事件的工具。 |
| Kernel time | 本專案量測報告中，同一 chip 各 operations 的 device 執行時間總和。 |
| Gap | 同一 chip 在一次 Metal Trace replay 內，兩個 device operations 之間的閒置時間。 |
| Metal Trace | TT-Metal 將 device 指令序列錄製到 buffer，後續以一次 host 呼叫重播的機制。 |
| Replay | 執行錄製的 TT-Metal 指令序列。 |
| Replay time | 未開 profiler 時，從 host 呼叫 blocking `execute_trace` 到返回的時間。 |
| TPOT | 每個 request 從第一個到最後一個輸出 token 的時間，除以第一個之後的輸出 tokens 數；彙整多個 requests 時註明統計方式。 |
| TTFT | 從 client 送出 request 到收到第一個輸出 token 的時間。 |
| Output throughput | 整個測試期間每秒總輸出 token 數。 |
| Microbenchmark | 專測特定路徑的計時。 |
| Warm-up | 正式量測前預先執行。 |
| Synthetic input | 人工產生的測試輸入。 |
| KV cache | 保存先前 tokens 的 attention keys 與 values；本實驗的 KV cache 位於各 chip 的 DRAM。 |
| Page table | 記錄每個 request 的 KV cache 位於哪些 pages 的表。 |
| Preemption | 暫停 request 讓出資源。 |
| Memory capture | buffer 配置與生命週期紀錄。 |
| NoC trace | chip 內資料流量紀錄。 |
| py-spy | Python 執行堆疊取樣工具。 |
| 分段計時 | Breakdown timing：把一個 decode step 拆成數段分別計時，例如 trace 執行與 logits readback。 |
| Baseline | 指定實驗的原始起點。 |
| 改動前 | 某項中途改動之前的受測版本，不稱為 baseline。 |
| 改動後 | 某項中途改動之後的受測版本。 |

## Profiling 與計時的補充

| Term | 定義 |
| --- | --- |
| Device firmware cycles | device 端 firmware 記錄的時脈週期。這次分析用各 operation 的 start/end cycles 計算一次 replay 內的 gaps。 |
| Ops report | 每個 device operation 一筆紀錄的量測報表，包含 chip、時間、tensor shapes 等欄位，例如 `OP TO OP LATENCY`。 |
| Profiler buffer | 用來儲存量測事件的空間。 |
| Marker | 標記事件的紀錄。Profiler buffer 空間不足時，markers 可能遺失。 |
| Decode latency | 一個 decode step 的耗時。引用數字時在前面寫出計時範圍，例如「以 `model.decode` 計時的 decode latency」，因為傳輸、host 分數處理與 token selection 不一定全部納入。 |
| Operation | 一次運算。 |
| Layer | 模型的一層。 |
| Shape | tensor 的維度大小。 |
| Dense MLP | 不依 expert routing 選擇子網路的 MLP。 |
| DRAM-bound | 效能主要受 DRAM 資料讀取或寫入限制。 |

計時範圍的可查證例子：

- `scripts/bench/decode.py` 直接呼叫 `model.decode`，不經 API 或 vLLM scheduler。計時包含在 host 建立輸入、把輸入從 host 複製到 device、device 運算、logits readback 與預設的 host softcap；prompt prefill、第一次 warm-up 與之後的 argmax 都不計入。[腳本](https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/scripts/bench/decode.py#L51-L68)
- 最新 audit 的 greedy decoding 計時另包含 token selection，不能與上一種時間直接互換。[Audit](experiments/2026-11-25-decode-readback-audit.md)
- Metal Trace replay 下的 `OP TO OP LATENCY` 包含 host 在 replays 之間的時間，不能直接當成 replay 內的 gap。[分析紀錄](experiments/2026-11-19-tt-scope-findings.md) · [分析腳本](../scripts/profile/trace_breakdown.py)

Synthetic input 若是 `torch.randn` 產生的 hidden states，應明寫來源，不以「真實 requests」或「random prompts」代稱。[測試腳本](https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/scripts/kernels/expert_stream_check.py#L82-L106)
