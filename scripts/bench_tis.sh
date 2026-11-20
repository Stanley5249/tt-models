#!/usr/bin/env bash
# Serving points built like tt-inference-server's release client, against a
# running `pixi run serve-gemma4`, so they can sit next to the official gates.
#
# Usage: pixi run bash scripts/bench_tis.sh <label> [point...]
# A point is c<concurrency>-<isl>-<osl>-n<prompts>[-sampled]. Defaults to the
# official c1 gates that fit a 32K server, plus our c4 point.
#
# Matches the release client (vLLM 0.13 `vllm bench serve`): openai-chat
# backend on /v1/chat/completions, random dataset, fixed ISL/OSL, ignore-eos,
# temperature 0 (0.13 hard-codes it; our 0.31 client does not, so it is passed),
# tput_user = 1000 / mean TPOT. Adds one discarded pass per point and three
# measured repetitions with seeds 0..2. "-sampled" drops --temperature, so the
# server's generation-config sampling (temperature 1, top_k 64) applies.
# Results: results/compare/<label>/<point>.{warmup,r0,r1,r2}.json
set -euo pipefail

label=${1:?label}
shift
points=("$@")
if [ ${#points[@]} -eq 0 ]; then
  points=(c1-128-128-n8 c1-2048-128-n4 c1-8192-128-n2 c4-128-128-n16)
fi
out="results/compare/$label"
mkdir -p "$out"

for p in "${points[@]}"; do
  IFS=- read -r conc isl osl n mode <<<"$p"
  conc=${conc#c}
  n=${n#n}
  for rep in warmup r0 r1 r2; do
    seed=$([ "$rep" = warmup ] && echo 100 || echo "${rep#r}")
    args=(--backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions
      --model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized --served-model-name gemma4
      --dataset-name random --random-input-len "$isl" --random-output-len "$osl" --random-range-ratio 0
      --ignore-eos --seed "$seed" --num-prompts "$n" --max-concurrency "$conc"
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
      --save-result --save-detailed --result-dir "$out" --result-filename "$p.$rep.json")
    if [ "${mode:-}" != sampled ]; then
      args+=(--temperature 0)
    fi
    vllm bench serve "${args[@]}" > "$out/$p.$rep.log" 2>&1
  done
  python - "$out" "$p" <<'EOF'
import json, sys
out, p = sys.argv[1:]
for rep in ("warmup", "r0", "r1", "r2"):
    r = json.load(open(f"{out}/{p}.{rep}.json"))
    print(p, rep, {
        "in_tok": round(r["total_input_tokens"] / r["num_prompts"], 1),
        "mean_ttft": round(r["mean_ttft_ms"], 1),
        "mean_tpot": round(r["mean_tpot_ms"], 2),
        "tput_user": round(1000 / r["mean_tpot_ms"], 2),
        "tput": round(r["output_throughput"], 2),
    })
EOF
done
