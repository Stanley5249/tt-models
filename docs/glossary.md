# Glossary

本頁是 tt-models 新文件中可重用術語的單一維護來源，不依賴 tt-home。報告首次使用術語時可提供簡短說明，完整定義與計時範圍集中在本頁。歷史實驗保留當時用語，引用時依本頁說明其含義。

數值與格式的例子以 Gemma 4 的 N300d 實驗為限，不代表所有模型或裝置。模型設定與量測脈絡見 [階段紀錄](experiments/2026-11-23-results-summary.md)與[模型設定](../src/tt_models/models/gemma4/config.py)。

| Term | 本文定義 |
| --- | --- |
| API／Checkpoint／Runtime | 程式呼叫服務的介面／儲存的模型權重與設定／執行模型所用的軟體環境。 |
| CPU governor | 控制 CPU 時脈策略的系統設定，performance 是其中一種策略。 |
| Buffer／Constant／Capture | 儲存資料的記憶體區域／預先建立並重用的固定資料／本文指錄製裝置指令，或 profiler 收集執行紀錄，依上下文區分。 |
| Exp underflow | 計算指數函數時，結果太小而在所用格式中成為零。 |
| N300d／Wormhole | 本實驗的 Tenstorrent 雙 chip 加速卡／chip 架構名稱。 |
| tt-scope／DRAM | 本實驗使用、整合裝置 profiling 與報表的工具／裝置上存放權重與 KV cache 等資料的記憶體。 |
| TTNN／TT-Metal | 裝置運算函式庫／底層執行與 kernel 程式設計層。 |
| vLLM／Scheduler | 模型服務框架／決定下一次模型執行處理哪些 requests 的元件。 |
| Request／Concurrency | 一次有獨立 prompt 與輸出的生成呼叫／同時未完成的 requests 數。 |
| Token／Prompt／Context | 模型文字單位／輸入內容／目前處理的 token 歷史。 |
| Vocabulary | 模型可輸出的全部 tokens，Gemma 4 有 262,144 個。Logits 對 vocabulary 中的每個 token 各有一個分數。 |
| Decode step／Active request／Batch size | 計算一次下一 token 分數／該次納入的 request／該次納入的數量。 |
| Prefill／Chunk／Prefix caching | 處理輸入 prompt／分段處理的區塊／重用相同輸入前綴的 KV cache。 |
| Tensor／Activation／Hidden state | 多維數值陣列／執行中產生的中間 tensor／一個 token 在模型某處的向量表示，本例為 2,816 components。 |
| Tile／Padding／Row-major | 本例 32×32 的資料區塊／為滿足儲存形狀補入的資料／各列連續儲存的排列。 |
| Expert／Router／MLP | 模型選用的子網路／選擇 experts 的運算／多層前饋網路。 |
| DRAM bank／Core／Kernel／Worker | 裝置記憶體分區／計算核心／裝置程式／由特定運算分配工作的核心。 |
| Operation fusion | 將多個計算合併成一個裝置 operation，減少中間資料搬運與啟動成本。 |
| Logits／Readback／LM head | 下一 token 的原始候選分數／傳回裝置結果／將 hidden state 轉成 logits 的輸出層。 |
| Greedy／Argmax／Sampling／Top-k／Seed | 選最高分 token／找最大值位置／按機率抽樣／選分數最高的 k 個候選／控制隨機過程的設定值。 |
| Logprobs／Softcap | token 機率的對數／壓縮 logits 幅度。數學上單調遞增，但 FP32 飽和時可能讓不同分數成為同分。 |
| Residual／RMSNorm | 保留並加入新輸出的 hidden state／按 root mean square 調整尺度的正規化。 |
| QKV／RoPE／SDPA／Output projection | attention 的 query、key、value／旋轉式位置編碼／scaled dot-product attention／attention 輸出的線性轉換。 |
| All-gather／All-reduce | 每顆 chip 收集其他 chips 的資料／結合各 chip 的資料，例如加總，再把結果交給各 chip。 |
| BF16／FP32／BFP4／BFP8 | 16-bit brain floating point／32-bit IEEE floating point／Tenstorrent 的 4-bit、8-bit block floating-point 儲存格式。 |
| Q4_0 | 每組 32 個權重搭配一個 scale 的 4-bit 格式；本專案保留實驗性計算路徑。 |
| QAT／FP32 reference／KL divergence | Quantization-aware training，訓練時納入量化影響／以 FP32 計算的對照模型／比較輸出機率分布的差異，本文簡寫 KL，越低越接近對照。 |
| Tracy／Profiler／Kernel time／Gap | 時間軸量測工具／記錄執行事件的工具／本專案量測報告中，同一 chip 各 operations 的裝置執行時間總和／運算間的裝置閒置。 |
| Metal Trace | TT-Metal 將裝置指令序列錄製到 buffer，後續以一次 host 呼叫重播的機制。 |
| Replay／Replay time | 執行錄製的 TT-Metal 指令序列／未開 profiler 時，從 host 呼叫 blocking `execute_trace` 到返回的時間。 |
| TPOT／TTFT／Output throughput | 首個輸出之後每 token 平均耗時／第一個輸出的等待時間／整個測試期間每秒總輸出 token 數。 |
| Microbenchmark／Warm-up／Synthetic input | 專測特定路徑的計時／正式量測前預先執行／人工產生的測試輸入。 |
| KV cache／Preemption／Recomputation | 保存 attention keys 與 values／暫停 request 讓出資源／重新計算先前內容以恢復執行。 |
| Memory capture／NoC trace／py-spy | buffer 配置與生命週期紀錄／chip 內資料流量紀錄／Python 執行堆疊取樣工具。 |
| Baseline／改動前與改動後 | 指定實驗的原始起點／某項中途改動的兩個受測版本，兩者不混用。 |

## Profiling 與計時的補充

| Term | 定義 |
| --- | --- |
| Device firmware cycles | 裝置端 firmware 記錄的時脈週期。這次分析用各 operation 的 start/end cycles 計算 replay span 與 gaps。 |
| Replay span | 同一 chip 的一次 replay，從最早 operation 開始至最晚 operation 結束的裝置時間，與 host 等待的 replay time 不同。 |
| Ops report | 每個 device operation 一列的量測報表，包含 chip、時間、tensor shapes 等欄位。 |
| Generic op | `ttnn.generic_op` 執行的自訂裝置運算；當時的報表將不同 kernels 都標為 `GenericOpDeviceOperation`。 |
| Profiler buffer／Marker | 用來儲存量測事件的空間／標記事件的紀錄；空間不足可能遺失紀錄。 |
| Decode latency | 一個 decode step 的耗時。引用數字時必須列出計時範圍，因為傳輸、host 分數處理與 token selection 不一定全部納入。 |
| Operation／Layer／Shape | 一次運算／模型的一層／tensor 的維度大小。 |
| Dense MLP／DRAM-bound | 不依 expert routing 選擇子網路的 MLP／效能主要受 DRAM 資料讀取或寫入限制。 |

計時範圍的可查證例子：

- `scripts/bench/decode.py` 直接呼叫 `model.decode`，不經 API 或 vLLM scheduler。計時包含 host input copy、device computation、readback 與預設 host softcap；prompt prefill、第一次 warm-up 與之後的 argmax 都不計入。[腳本](../scripts/bench/decode.py#L51-L68)
- 最新 audit 的 greedy step 另包含 token selection，不能與上一種時間直接互換。[Audit](experiments/2026-11-25-decode-readback-audit.md)
- Metal Trace replay 下的 `OP TO OP LATENCY` 包含 host 在 replays 之間的時間，不能直接當成 replay 內的 gap。[分析紀錄](experiments/2026-11-19-tt-scope-findings.md) · [分析腳本](../scripts/profile/trace_breakdown.py)

Synthetic input 若是 `torch.randn` 產生的 hidden states，應明寫來源，不以「真實 requests」或「random prompts」代稱。[測試腳本](../scripts/kernels/expert_stream_check.py#L82-L106)
