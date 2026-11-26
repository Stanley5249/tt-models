# 使用 tt-scope 分析的 Gemma 4 推論最佳化：附錄

[回主文](gemma4-optimization-with-tt-scope.md)

## tt-scope 操作細節

Batch size 1 decode 最佳化的紀錄包含九次 Tracy captures，batch size 1，每次三個 decode steps。Agent 以報表中的 device firmware cycles 分析每次 replay，再按 layer 內的位置整理 operations。[分析紀錄][findings]

Batch size 4 調查當時執行的命令：

```sh
pixi run tt-scope tracy scripts/profile_workload.py --programs 8000 -- --batch 4 --decode-steps 2 --trace
```

現行腳本已移到 `scripts/profile/workload.py`。2026-11-22 起的 batch size 4 captures 要用 `--programs 20000`，8,000 與 12,000 時有 cores 遺失 markers，所以不能把同一個 buffer 大小套用到所有工作。[原始命令][rawbatch] [現行腳本][workload] [2026-11-22 capture][stack]

當時 tt-scope 的平面報表沒有提供全部分組，operation 數改變時，`trace_breakdown.py` 的分組也要跟著更新。自訂 `ttnn.generic_op` 在報表中名稱相同，要再以 tensor shape 與 core count 辨認。最新調查也錄了 Memory capture 和 NoC trace，但現有紀錄無法說明主文哪一項加速來自這兩種 capture。[分析腳本][breakdown] [2026-11-23 調查][round3] [Capture 與分段測試][reassessment]

## 服務穩定性改動

最後三項是程式審查找到的正確性修正，已納入最終程式，沒有獨立的效能數字。

- **Prefill warm-up**：新輸入長度的服務計時顯示首次編譯成本。將長度分組並預先編譯後，這組新長度、只生成一個 token 的 chat request 平均全程耗時 **3.4–4.9 → 1.17–1.37 秒**，不代表所有新 shape 都已涵蓋。[紀錄][buckets]
- **KV cache ownership**：原本 vLLM 只以 page 管理 5 個 global layers 的 KV cache，25 個 sliding layers 由 adapter（`vllm_adapter.py`）各自保存，所以 prefix caching、chunked prefill 與 preemption 無法涵蓋全部 layers。全部改由 vLLM 管理後，測試中可重用 6,144-token prefix，256 個新輸入 tokens 的 TTFT 為 **323 ms**。這個 prefix 長度沒有改動前的配對量測，不計算加速比。[紀錄][kv]
- **Prefill/decode scheduling**：8,192-token prompt 在其他 request 輸出途中到達時，該 request 的輸出會停頓。原本每次 decode 前跑兩個 2,048-token prefill chunks，改為一個後，進行中 request 的最長停頓 **3.38 → 1.72 秒**，8K prompt 的 TTFT **6.53 → 6.56 秒**，各三次測試。[紀錄][interleave]
- **Trace allocation safety**：在 server 外依原順序重播服務輸入，重現了 trace capture 之後才配置的 buffers 被 replay 覆寫的問題。改為提前配置 constants 與 decode inputs，並移除 prefill 中 `ttnn.all_reduce` 依 shape 快取的 semaphores，修復已重現的錯誤輸出與卡住問題，不主張加速。[紀錄][safety]
- **Greedy kernel initialization**：Audit 的程式檢查發現 readback program 在首次使用時才建立。改在 capture 前建立，並保留初始化修正。公開 audit 記錄新增的 regression test 在舊版失敗。vLLM 的 warm-up 原本就會先建立這個 program，所以服務沒有遇到這個問題，不主張加速。[紀錄][audit]
- **RoPE scratch 與執行緒設定**：程式審查找到 q/k RoPE 暫存空間越界與 Torch 執行緒數競態。修正邊界與設定方式。[紀錄][round3]
- **Q4_0 routing**：程式審查發現 prefill routing 缺少 decode kernels 需要的 expert count。改用 decode router。[紀錄][round3]
- **SDPA page-table bounds**：程式審查發現奇數寬度的 page table 可能遭越界讀取。限制 chunk 的使用條件。[紀錄][round3]

## 最新服務結果

下表是最新 audit 中三項改動的合計效果：greedy 候選 readback、RoPE input reuse 和 small all-gather。數字不是各項的獨立收益，也不是與最早實作比較。

量測條件為同一張 N300d、同日、CPU governor 設為 performance，128 input／128 output tokens，使用 chat endpoint 與 greedy，丟棄 warm-up，每點取三個 seeds 的平均。改動前是 `f87908f`，改動後是經 audit 的程式。[設定與原始資料][audit]

| Concurrent requests | TPOT 改動前 → 後 | Output throughput 改動前 → 後 |
| ------------------: | ---------------: | ----------------------------: |
|                   1 | 16.49 → 16.09 ms |        54.95 → 56.16 tokens/s |
|                   4 | 23.67 → 22.41 ms |      144.05 → 151.35 tokens/s |
|                   8 | 31.23 → 28.92 ms |      212.13 → 226.17 tokens/s |
|                  16 | 44.32 → 39.36 ms |      288.77 → 316.76 tokens/s |

Output throughput 以整個測試耗時為分母，包含 prefill，所以不能用 concurrent requests × 1,000／TPOT 推算。任一 active request 使用 sampling 或要求 logprobs 時，整批都讀回完整 logits，候選 readback 不適用。[Audit][audit]

## 未採用的最佳化

- **Lower-precision LM head**：權重搬運仍有成本，所以嘗試 BFP4。在改用自訂 LM head kernels 之前，batch size 1 單步 **17.05 → 15.95 ms**，但三個配對 seeds 的 KL 都變差，平均 **0.4436 → 0.4572**。預設保留 BFP8，BFP4 仍可手動選用。[紀錄][round3]
- **Async all-reduce／更多 links**：Microbenchmark 嘗試 `all_reduce_async` 與 `num_links=2` 兩種跨 chip 傳輸，兩者都曾讓裝置卡住，未保留。[紀錄][r2]
- **合併兩個權重讀取 kernels**：配對計時嘗試合併兩個 DRAM-bound 路徑以減少 operations，結果反而變慢，未保留。[紀錄][r2]
- **Full device sampling**：分段計時與 top-k prototype 評估在裝置上產生 candidates，當時收益不足以支持完整實作，沒有建置。它和主文的 greedy readback 是不同的改動。[紀錄][sampling]

## 正確性與驗證限制

2026-11-17 與 2026-11-19 的實驗用 FP32 reference 檢查模型品質。[初始品質][r1] [單請求品質][r2] 最新 audit 確認受測的完整 logits hashes 一致，6,048 次 greedy token 比對一致。128 筆單獨送出的服務回覆與改動前逐 token 一致。同條件的並行 chat 有 **78／80** 筆新舊回覆一致，並行 raw-text 測試在兩版各自重跑時也會變動。並行與單獨執行可能因 batch 的數值捨入而不同，所以這些測試不能證明所有輸入都完全等價。[最新 audit][audit]

最新 audit 未重測約 300 個 token 位置之後的速度，也未重跑 preemption。獨立程式審查另指出兩項待裝置驗證的風險：批次 prefill 可能產生 warm-up 未涵蓋的新 shape，router 在極端 exp underflow 時可能產生重複的 expert IDs。這兩項尚未在目前 checkpoint 的裝置執行中重現，不能視為已修復。[批次分組][groupcode] [Warm-up code][warmupcode] [Router code][routercode]

這些數字沒有重新在硬體上執行，也不能單獨量化 tt-scope 的效果。

[findings]: ../experiments/2026-11-19-tt-scope-findings.md
[breakdown]: ../../scripts/profile/trace_breakdown.py
[workload]: ../../scripts/profile/workload.py
[rawbatch]: ../experiments/2026-11-21-batched-expert-streams/decode-b4-before.txt
[reassessment]: ../experiments/2026-11-25-gemma4-decode-readback.md
[r1]: ../experiments/2026-11-17-decode-round-1/report.md
[r2]: ../experiments/2026-11-19-decode-round-2/report.md
[round3]: ../experiments/2026-11-23-decode-serving-round-3.md
[audit]: ../experiments/2026-11-25-decode-readback-audit.md
[stack]: ../experiments/2026-11-22-shared-stack.md
[buckets]: ../experiments/2026-11-21-prefill-length-buckets.md
[kv]: ../experiments/2026-11-21-hybrid-kv-cache.md
[interleave]: ../experiments/2026-11-21-decode-interleave.md
[safety]: ../experiments/2026-11-21-trace-safety.md
[sampling]: ../experiments/2026-11-21-device-sampling.md
[warmupcode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L457
[routercode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/kernels/router_writer.cpp#L29
[groupcode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L350
