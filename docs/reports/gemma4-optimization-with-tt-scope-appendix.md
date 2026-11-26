# 使用 tt-scope 分析的 Gemma 4 推論最佳化：附錄

[回主文](gemma4-optimization-with-tt-scope.md)

## tt-scope 操作細節

2026-11-19 用 tt-scope v0.4.0 對 batch size 1 錄製 9 次 Tracy captures，每次 3 個 decode steps，再把 operations 依 chip 與所屬 layer 分組。在 Metal Trace replay 下，報表的 `OP TO OP LATENCY` 包含 host 在 replays 之間的時間。這一欄加總約 395 ms，但整個 decode step 約 34 ms。Agent 改用 device firmware 的 start/end cycles 重算，得到 replay 內的 gaps 合計約 0.8 ms，之後改查各 kernels 的耗時。[分析紀錄][findings] [分析腳本][breakdown]

Agent 依未開 profiler 時改動前與改動後的計時或服務測試，決定是否採用改動。

Batch size 4 調查當時執行的命令：

```sh
pixi run tt-scope tracy scripts/profile_workload.py --programs 8000 -- --batch 4 --decode-steps 2 --trace
```

現行腳本已移到 `scripts/profile/workload.py`。`--programs` 設定 profiler buffer 能容納的 programs 數。2026-11-22 的 batch size 4 Tracy capture 在 8,000 與 12,000 時都有 cores 遺失 markers，所以這組 capture 改用 `--programs 20000`。[原始命令][rawbatch] [現行腳本][workload] [2026-11-22 capture][stack]

當時 tt-scope 的 ops report 沒有提供全部分組，operation 數改變時，`trace_breakdown.py` 的分組也要跟著更新。自訂 `ttnn.generic_op` 在報表中名稱相同，要再以 tensor shape 與 core count 辨認。2026-11-25 的調查也錄了 Memory capture 和 NoC trace，但現有紀錄無法說明主文哪一項加速來自這兩種 capture。[分析腳本][breakdown] [2026-11-23 調查][round3] [Capture 與分段計時][reassessment]

## 各項的量測條件

- **Runtime**：2026-11-22 起，runtime 改用 tt-scope 0.5.0 提供的版本，TTNN 從 0.81.1 換成 0.82.0，tt-scope 從 0.4.0 換成 0.5.0，兩邊都使用 vLLM 0.32.0。以[換版前的程式][0d46091]比較，換版後 concurrency 1 的服務 TPOT 為 18.10–18.16 → 17.46–17.47 ms，這是軟體版本變更的效果，不是最佳化。[紀錄][stack]
- **Logits readback 不傳 padding rows**：2026-11-17 的實驗。[紀錄][r1]
- **按 DRAM bank 讀取 expert weights**：2026-11-19 先用於 batch size 1，replay time 為 30.1 → 24.1 ms。[Commit][775e4ec] [紀錄][r2] 2026-11-21 延伸到 batch size 2–4。[Commit][0e65f24] Batch size 4 的 Tracy capture 中，`ttnn.sparse_matmul` 占 chip 0 kernel time 的 20.5 ms，85 GB/s 是以 32 個 expert slots 估算。改動後 gate/up 的 reader kernels 每次讀取 6 KB。比較改動前後的 Tracy capture，chip 0 的 kernel time 合計為 50.86 → 31.45 ms，gate/up 的讀取速度約 204 GB/s，down projection 約 226 GB/s。同一 commit 也讓 dense MLP 改用這組 kernels，所以主文未開 profiler 的 decode latency 涵蓋兩者。[改動前 capture][rawbatch] [改動後 capture][rawbatchafter] [紀錄][batched]
- **Hidden state 改放進 3 個 tiles**：2026-11-19 先用於 batch size 1，2026-11-21 延伸到 batch size 2–4。[Batch size 1 commit][dab3331] [Batch size 2–4 commit][333c5e8] Batch size 4 的 Tracy capture 中，改動前另有 60 次 all-gather 約占 3.3 ms。比較改動前後的 Tracy capture，chip 0 的 operations 為 1,778 → 784，kernel time 合計為 30.55 → 24.83 ms。Concurrency 4 的 output throughput 為 96.9 → 111.3 tokens/s。主文的 decode latency 未開 profiler，它與服務 TPOT 的比較都包含[自訂 router 延伸到 batch size 2–4][7a3ccaf] 的改動，residual 也由 BF16 改為 FP32，所以兩組數字不能全部歸因於 hidden state 的排列或 fusion。[改動前 capture][rawboundary] [改動後 capture][rawboundaryafter] [紀錄][boundaries]
- **每個 expert 的 weights 只讀一次**：2026-11-23 的實驗。[Commit][0499452] Distinct experts 數來自 `expert_stream_check.py` 的 kernel test。這個 test 以 `torch.randn` 產生 hidden states，經 RMSNorm 後使用實際模型權重，所以結果不代表實際 prompts。Batch size 4 的 32 次 expert 選擇涉及 29–30 個 distinct experts。[測試腳本][expertcheck] 同一 commit 也把 `STREAM_ROWS` 從 4 擴至 16，router 改按 expert tiles 分配工作，batch size 超過 10 時 tail op 減少分給每個 request 的 cores，batch size 9–15 時 attention 以 `permute` 串接 heads，所以 batch size 4 的 decode latency 差距不全來自共用讀取。Concurrency 16 時，每個 request 的 TPOT 由 30.08 升到 51.54 ms。[紀錄][shared]
- **多個 requests 的 prefill 放進同一個 batch**：2026-11-23 的實驗。[Commit][c375703] Adapter 只把 prefill 片段不超過 `PREFILL_CHUNK` 個 tokens 的 requests 分組，每組合計的 padded rows 不超過 `PREFILL_CHUNK`，較長的片段各自執行。2 個各 1,500 tokens 的 prompts，prefill 耗時只有 1.99 → 1.88 秒。主文的 TTFT 比較也包含 [router top-k 分散到多核心][7c565e5] 的改動。[紀錄][round3]
- **Greedy decoding 只傳回每段的最高 logit**：[Commit][bf41a37] 2026-11-23 對 16 個 concurrent requests 的 server 錄製的 `py-spy` stack samples 中，`decode_forward` 內有 89% 的 samples 包含 `LmHead.read`，這段時間包含等待 device 完成運算與讀回 logits。[紀錄][round3] 2026-11-25 的 audit 在[加入 small all-gather 後的程式][e87dbd9]上交錯測試三種 token selection 方法。主文的 41.24 ms 是 host 讀回未經 softcap 的完整 logits 後執行 PyTorch argmax，改用 NumPy argmax 則為 39.54 ms。單一 thread 的 CPU 測試中，對 16 個 requests 的 logits 執行 PyTorch argmax 需 2.518 ms。[分段計時與 CPU 測試][reassessment]三種方法逐 request、逐 decode step 共比對 6,048 次，選出的 token 都一致。Concurrency 16 的服務 TPOT 為 44.32 → 39.36 ms，這組比較也包含 RoPE input reuse 與 small all-gather，見下方「2026-11-25 audit 的服務結果」。[Audit][audit] [read_greedy][greedycode] [啟用條件][adaptercode]
- **Router fusion**：2026-11-17 的實驗。[Commit][f32639d] [紀錄][r1]
- **Expert gate/up padding**：2026-11-17 的實驗，與 router fusion 在同一個 commit。[Commit][f32639d] [紀錄][r1]
- **QKV processing**：2026-11-17 的實驗。[Commit][ea2e6a0] 另一個 commit 讓 QKV 的後處理保持 FP32，4 段各 128 個 decode steps 的 KL 為 0.394 → 0.340。[Commit][1573ed6] [紀錄][r1]
- **Prefill 先按 expert 整理 tokens**：2026-11-17 的實驗。[Commit][b85f4ab] 8,192-token prefill 為 20.8 → 5.5 秒。512-token 的 KL 為 0.351 → 0.355，在量測雜訊範圍內。[紀錄][r1]
- **只計算 active requests**：2026-11-17 的實驗。[Commit][807f056] [紀錄][r1]
- **Top-k sampler**：2026-11-17 的實驗。[Commit][99cc25a] Adapter 的 top-k sampling 約需 1 ms。Commit 當時，沒有設定 top-k 的 requests 也交給 vLLM sampler，本文描述的程式改以 1,024 個 candidates 處理這類 requests。Concurrency 4 的 output throughput 為 28.6 → 60.9 tokens/s。[紀錄][r1]
- **All-reduce 改寫**：2026-11-19 的實驗。[Commit][2104d70] Microbenchmark 中，改寫後處理 2,816 個值的 padded row 需 34 µs。[紀錄][r2]
- **Dense MLP 按 DRAM bank 讀取**：2026-11-19 的實驗。[Commit][350a68c] [紀錄][r2]
- **自訂 router**：2026-11-19 用於 batch size 1，每 layer 為 71 → 36 µs，batch size 1 的 replay time 為 17.7 → 16.7 ms。[Commit][fd5a079] [紀錄][r2] 2026-11-21 延伸到 batch size 2–4。改動前 batch size 2–4 仍用 `topk_router_gpt`，每 layer 69–72 µs，另有四個 reshape 與 typecast operations。改動後 batch size 3–4 每 layer 為 54–68 µs。[Commit][7a3ccaf] [紀錄][multirouter]
- **只對選出的 tokens 做 softcap**：2026-11-21 的實驗。[Commit][8aa505e] 要求 logprobs 或改走 vLLM host sampler 時，host 仍對完整 logits 做 softcap。Concurrency 1 的服務 TPOT 為 18.46 → 18.14 ms。[紀錄][softcap]
- **Q/K RMSNorm、RoPE 與 host 輸入**：2026-11-23 的實驗。[Commit][87f99ff] Fuse 後的 operation 每 layer 需 28–47 µs。[紀錄][round3]
- **Router top-k 分散到多核心**：2026-11-23 的實驗。[Commit][7c565e5] Batch size 16 時，router operation 為 145 → 69 µs。[紀錄][round3]
- **SDPA K chunk size**：2026-11-23 的實驗。[Commit][0875726] [Bounds 修正 commit][4750c35] [紀錄][round3]
- **各 chip 依自己的 DRAM bank 位置選擇讀取 cores**：2026-11-19 的紀錄顯示，沿用 chip 0 的讀取 cores 配置時 chip 1 較慢。[紀錄][r2] 2026-11-21 交換兩顆 chips 的配置，確認原因是兩顆 chips 的 harvesting 不同，當時的改動收益太小而撤回。[紀錄][placement] 2026-11-23 再分別修正 LM head 與 expert kernels。 LM head 改用依各 chip 配置的讀取 kernels 後，chip 1 的 LM head operation 為 2.41 → 1.68 ms。[LM head commit][41ef565] Expert kernels 也改成依各 chip 配置後，batch size 4 以 `model.decode` 計時的 decode latency 為 23.21 → 22.78 ms。[Expert commit][ed12310] [紀錄][round3]
- **Attention output projection 直接輸出 3-tile hidden state**：2026-11-23 的實驗。[Commit][1b4b150] [紀錄][round3]
- **RoPE input reuse**：2026-11-25 的實驗。[Commit][9d2b250] [紀錄][reassessment]
- **Small all-gather**：2026-11-25 的實驗。[Commit][e87dbd9] Microbenchmark 中，3-tile all-gather 的耗時為 10.94 → 7.97 µs，6-tile all-gather 為 12.03 → 10.28 µs。[紀錄][reassessment]

## 服務穩定性改動

以下只有 prefill warm-up 與 prefill/decode scheduling 有改動前後的速度量測，其餘是正確性修正，已納入最終程式。

- **Prefill warm-up**：服務計時顯示，新輸入長度的第一個 request 要等 device programs 編譯。把 prefill 長度分到固定的 buckets，並預先編譯每個 bucket 後，8 個新長度、只生成一個 token 的 chat request 平均全程耗時 **3.4–4.9 → 1.17–1.37 秒**。Expert chunk 數依 routing 而變，新的 chunk 數仍要建置一次 program，約 0.4 秒。[紀錄][buckets]
- **KV cache 全部改由 vLLM 管理**：原本 vLLM 只以 pages 管理 5 個 global attention layers 的 KV cache。其餘 25 個 sliding-window attention layers 的 K/V 也放在 device DRAM。`vllm_adapter.py` 為每個 batch slot 管理一個 1,024-token ring buffer，所以 prefix caching、chunked prefill 與 preemption 無法涵蓋全部 layers。全部改由 vLLM 管理後，測試中可重用 6,144-token prefix，256 個新輸入 tokens 的 TTFT 為 **323 ms**。這個 prefix 長度沒有改動前的量測，不計算加速比。[紀錄][kv]
- **Prefill/decode scheduling**：8,192-token prompt 在另一個 request 輸出途中到達時，進行中 request 的輸出會停頓。原本每次 decode 前跑兩個 2,048-token prefill chunks，改為只跑一個後，進行中 request 的最長停頓 **3.38 → 1.72 秒**，8,192-token prompt 的 TTFT **6.53 → 6.56 秒**，各三次測試。[紀錄][interleave]
- **Trace allocation safety**：Agent 在 vLLM server 外，依 server 的呼叫順序，用匯出的 prefill 與 decode inputs 執行[改動前的程式][2a525b3]，重現了 replay 覆寫 device buffers 的問題。這些 buffers 是在 Metal Trace capture 之後才配置的。改為提前配置 constants 與 decode inputs，並移除 prefill 中 `ttnn.all_reduce` 依 shape 快取的 semaphores，修復已重現的錯誤輸出與卡住問題。[紀錄][safety]
- **Greedy kernel initialization**：Audit 的程式檢查發現 readback program 在首次使用時才建立。改在 Metal Trace capture 前建立，新增的 regression test 在舊版失敗。vLLM 的 warm-up 原本就會先建立這個 program，所以服務沒有遇到這個問題。[紀錄][audit]
- **Q/K RoPE scratch 與 PyTorch thread 數**：Code review 發現，Q/K RoPE scratch 有越界寫入，PyTorch thread 數的設定也有競態。兩者都已修正。[紀錄][round3]
- **Q4_0 routing**：Code review 發現，Q4_0 prefill 原用 `topk_router_gpt`，它的 expert IDs 輸出沒有附上這次選中的 distinct experts 數量，而共用的權重讀取 kernels 需要這個數量。改用 decode router。[紀錄][round3]
- **SDPA page-table bounds**：Code review 發現，KV pool 較小使 page table 寬度為奇數時，SDPA 的 128-position chunk 可能讀到 page table 之外。[修正][4750c35]後只在 page table entries 數乘上 KV block size 可被 128 整除時才讀 128 個 positions。[紀錄][round3]

## 2026-11-25 audit 的服務結果

下表是 2026-11-25 audit 中三項改動的合計效果：greedy readback、RoPE input reuse 和 small all-gather。數字不是各項的獨立收益。

量測條件為 CPU governor 設為 performance、128 input／128 output tokens、chat endpoint 與 greedy，丟棄一次 warm-up，每點取三個 seeds 的平均。改動前為 [audit 前的程式][f87908f]，改動後為 audit 後的程式，各啟動一次 server，先測改動後。[設定與原始資料][audit]

| Concurrent requests | TPOT 改動前 → 後 | Output throughput 改動前 → 後 |
| ------------------: | ---------------: | ----------------------------: |
|                   1 | 16.49 → 16.09 ms |        54.95 → 56.16 tokens/s |
|                   4 | 23.67 → 22.41 ms |      144.05 → 151.35 tokens/s |
|                   8 | 31.23 → 28.92 ms |      212.13 → 226.17 tokens/s |
|                  16 | 44.32 → 39.36 ms |      288.77 → 316.76 tokens/s |

任一 active request 使用 sampling 或要求 logprobs 時，整批都讀回完整 logits，greedy readback 不適用。[Audit][audit]

## 未採用的最佳化

- **Lower-precision LM head**：LM head 每個 decode step 都要從 DRAM 讀取整個 BFP8 權重，所以嘗試把權重改為 BFP4。在改用自訂 LM head kernels 之前，batch size 1 以 `model.decode` 計時的 decode latency 為 **17.05 → 15.95 ms**，但在相同的三個 seeds 上，KL 都比 BFP8 差，平均 **0.4436 → 0.4572**。預設保留 BFP8，BFP4 仍可手動選用。[紀錄][round3]
- **Async all-reduce／更多 links**：Microbenchmark 嘗試 `all_reduce_async` 與 `num_links=2` 兩種跨 chip 傳輸，兩者都曾讓 device 卡住，未保留。[紀錄][r2]
- **兩個權重讀取 kernels 放進同一個 program**：嘗試把兩個 DRAM-bound 的讀取 kernels 放進同一個 program 以減少 operations。放進同一個 program 後比依序執行慢，未保留。[紀錄][r2]
- **在 device 上完成 sampling**：2026-11-21 評估在 device 上完成 sampling，讓 async decode 把 host 工作與下一個 decode step 重疊。估計 batch size 1 每個 decode step 只省約 0.6–1 ms，卻需要保留 per-core top-k 的自訂 LM head kernel，以及放在 device 上的 embedding，所以沒有建置。[紀錄][sampling]

## 正確性與驗證限制

2026-11-17 與 2026-11-19 的實驗用 FP32 reference 檢查模型品質。[初始品質][r1] [Batch size 1 品質][r2] 2026-11-25 audit 確認受測的完整 logits hashes 一致。128 筆單獨送出的服務回覆與改動前逐 token 一致。每批 16 個 concurrent chat requests，共 80 筆回覆中有 **78** 筆在改動前與改動後一致。Concurrent raw-text 測試在兩版各自重跑時也會變動。Concurrent 與單獨執行可能因 batch 的數值捨入而不同，所以這些測試不能證明所有輸入都完全等價。[2026-11-25 audit][audit]

2026-11-25 audit 未重測約 300 個 token 位置之後的速度，也未重跑 preemption 情境。從[本文描述的程式][head]可看出兩項風險。Batched prefill 可能產生單一 request warm-up 未涵蓋的 shape。Exp underflow 使正值少於 top-k 個時，router 可能留下重複的 expert IDs。本文引用的紀錄沒有在這個版本上重現這兩種情況，所以兩者仍待 device 驗證。[Batched prefill 分組][groupcode] [Warm-up code][warmupcode] [Router code][routercode]

現有量測無法單獨估計 tt-scope 對效能的影響。

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
[rawbatchafter]: ../experiments/2026-11-21-batched-expert-streams/decode-b4-after.txt
[batched]: ../experiments/2026-11-21-batched-expert-streams.md
[rawboundary]: ../experiments/2026-11-21-compact-boundaries-multi-row/decode-b4-main.txt
[rawboundaryafter]: ../experiments/2026-11-21-compact-boundaries-multi-row/decode-b4-compact.txt
[boundaries]: ../experiments/2026-11-21-compact-boundaries-multi-row.md
[shared]: ../experiments/2026-11-23-shared-expert-reads.md
[expertcheck]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/scripts/kernels/expert_stream_check.py#L82-L106
[multirouter]: ../experiments/2026-11-21-router-op-multi-row.md
[softcap]: ../experiments/2026-11-21-lazy-softcap.md
[head]: https://github.com/Stanley5249/tt-models/commit/342e751a47665fda8e52bf407cb13f7f544d10a1
[775e4ec]: https://github.com/Stanley5249/tt-models/commit/775e4ec
[0e65f24]: https://github.com/Stanley5249/tt-models/commit/0e65f24
[dab3331]: https://github.com/Stanley5249/tt-models/commit/dab3331
[333c5e8]: https://github.com/Stanley5249/tt-models/commit/333c5e8
[0499452]: https://github.com/Stanley5249/tt-models/commit/0499452
[c375703]: https://github.com/Stanley5249/tt-models/commit/c375703
[bf41a37]: https://github.com/Stanley5249/tt-models/commit/bf41a37
[f32639d]: https://github.com/Stanley5249/tt-models/commit/f32639d
[ea2e6a0]: https://github.com/Stanley5249/tt-models/commit/ea2e6a0
[1573ed6]: https://github.com/Stanley5249/tt-models/commit/1573ed6
[b85f4ab]: https://github.com/Stanley5249/tt-models/commit/b85f4ab
[807f056]: https://github.com/Stanley5249/tt-models/commit/807f056
[99cc25a]: https://github.com/Stanley5249/tt-models/commit/99cc25a
[2104d70]: https://github.com/Stanley5249/tt-models/commit/2104d70
[350a68c]: https://github.com/Stanley5249/tt-models/commit/350a68c
[fd5a079]: https://github.com/Stanley5249/tt-models/commit/fd5a079
[7a3ccaf]: https://github.com/Stanley5249/tt-models/commit/7a3ccaf
[8aa505e]: https://github.com/Stanley5249/tt-models/commit/8aa505e
[87f99ff]: https://github.com/Stanley5249/tt-models/commit/87f99ff
[7c565e5]: https://github.com/Stanley5249/tt-models/commit/7c565e5
[0875726]: https://github.com/Stanley5249/tt-models/commit/0875726
[4750c35]: https://github.com/Stanley5249/tt-models/commit/4750c35
[41ef565]: https://github.com/Stanley5249/tt-models/commit/41ef565
[ed12310]: https://github.com/Stanley5249/tt-models/commit/ed12310
[1b4b150]: https://github.com/Stanley5249/tt-models/commit/1b4b150
[9d2b250]: https://github.com/Stanley5249/tt-models/commit/9d2b250
[e87dbd9]: https://github.com/Stanley5249/tt-models/commit/e87dbd9
[f87908f]: https://github.com/Stanley5249/tt-models/commit/f87908f
[0d46091]: https://github.com/Stanley5249/tt-models/commit/0d46091
[2a525b3]: https://github.com/Stanley5249/tt-models/commit/2a525b3
[placement]: ../experiments/2026-11-21-per-chip-placement.md
[greedycode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/lm_head.py#L226
[adaptercode]: https://github.com/Stanley5249/tt-models/blob/342e751a47665fda8e52bf407cb13f7f544d10a1/src/tt_models/models/gemma4/vllm_adapter.py#L400
