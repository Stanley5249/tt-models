# Gemma 4 與 Pi 驗證報告

2026-11-19 · Tenstorrent N300 · 基準 commit `a527dad`，含工作目錄既有的 `pixi.toml` 修改。

## 測試結果

首次啟動卡在裝置 0 的 ARC 韌體初始化。等候超過 130 秒後，API 仍未就緒。經授權停止測試程序後，使用 `tt-smi --offline --reset /dev/tenstorrent/0` 重設該卡。用同一份設定重啟後，`/v1/models` 成功列出 `gemma4`，上下文上限為 32,768 tokens。首次記錄末尾的 `KeyboardInterrupt: terminated` 由停止程序觸發。ARC 韌體卡住的原因仍未確認。 首次啟動記錄 (`logs/serve-pi-workflow.WehX8d.log:44`, local only)

Pi 透過 `tt-vllm/gemma4` 讀取暫存檔中的隨機值，修改另一個檔案，再用 Bash 執行驗證腳本。工具順序為 `read → read → edit → bash`，四次都成功。獨立的檔案比對通過，最終回覆也符合驗證腳本的輸出。Pi 收到 5 個非空的 Thinking 區塊。原本的斷言只允許一次 `read`，因此回報失敗。重新檢查同一批事件後，確認多一次讀檔仍符合流程。5 次聊天請求皆回傳 HTTP 200。 重試記錄 (`logs/serve-pi-workflow.rOTdhO.log:683`, local only)

## 重現的警告

| 階段 | 觀察 | 交接重點 |
| --- | --- | --- |
| 暖機 | HiFi4 搭配 FP32 累加出現 1 次精度警告。 | 找出使用該組合的算子，用相同輸入比較 HiFi3 和參考輸出。Pi 工具測試未涵蓋數值精度。 |
| 暖機 | trace 啟用時配置裝置緩衝區，出現 1 次可能被覆寫的警告。 | 用 trace allocation tracker 確認緩衝區是否跨 trace 重播存活。本次未驗證。 |
| 推論 | semaphore 回退到一般 L1 共 10 次。fabric 封包大小不理想的警告出現 1 次。 | v0.81.1-conda.8 的 all-gather 路徑會在 `L1_SMALL` bank 大小非零時優先使用該區域。TT plugin 可傳入 `tt.l1_small_size`。尚未做 A/B 測試，本次也未看到 L1 配置失敗。 |
| 關機 | 程序管理器記錄會強制清理 EngineCore，最後完成關機。 | 若 ARC 卡住再現，保留前一輪關機與下一輪啟動記錄。兩起事件的關聯尚未確認。 |

啟動時也出現主機板辨識回退、缺少 PGD、無法調整執行緒優先權、自訂 scheduler、Inductor 已停用，以及 checkpoint 覆寫預設取樣參數等警告。重試記錄中沒有 HTTP 4xx、5xx 或 traceback。 啟動記錄 (`logs/serve-pi-workflow.rOTdhO.log:19`, local only) · 暖機 (`logs/serve-pi-workflow.rOTdhO.log:634`, local only) · 推論 (`logs/serve-pi-workflow.rOTdhO.log:685`, local only) · 關機 (`logs/serve-pi-workflow.rOTdhO.log:718`, local only)

## 目前的 adapter 限制

| 能力 | 目前行為 |
| --- | --- |
| Chunked prefill | `prefill_forward()` 拒絕非零的 `start_pos`。停用分段後，TT plugin 把 `max_num_batched_tokens` 從 2,048 調到 32,768。 |
| Prefix caching | vLLM 管理 5 層全域注意力的分頁 KV。其餘 25 層滑動視窗 KV 綁在各個 slot 的環形快取，尚未實作跨請求的狀態恢復。 |
| Async decode | adapter 使用舊版 decode 輸入更新介面。此次執行使用同步排程，沒有 decode 重疊。 |
| 取樣 | `supports_sample_on_device=True` 會走 adapter 的主機端 top-k 取樣器，`max_device_top_k` 為 1,024。裝置端 penalties 和 grammar 均未宣告支援。 |

前三項是本地 Gemma 4 adapter 的能力宣告，由 TT plugin 套用。若要支援 prefix caching，先驗證完整 prefill 和分段 prefill 的 logits 在容許誤差內一致。測資應跨過滑動視窗和 KV page 邊界。 能力宣告 (`gemma4_tt/vllm_model.py:70` at `a527dad`) · Prefill 入口 (`gemma4_tt/vllm_model.py:130` at `a527dad`) · Decode 記錄 (`logs/serve-pi-workflow.rOTdhO.log:689`, local only)

## 重現方式

執行 `pixi run --locked serve-gemma4`，等 `http://127.0.0.1:8000/v1/models` 列出 `gemma4`。用 `pi --offline --no-session --model tt-vllm/gemma4 --tools read,edit,bash --mode json -p` 重跑讀檔、修改和比較流程。驗證工具的相對順序與結果，容許額外的成功讀檔。另用長文請求測試輸出長度，記錄生成額度、用量及停止原因。此次工具流程未重現先前的 `finish_reason: length`。

測試服務已停止，連接埠 8000 已釋放，暫存測試檔已清除。兩份伺服器記錄留在本機 `logs/`，該目錄受 Git 忽略。測試期間沒有修改程式碼或模型設定。
