#!/usr/bin/env bash
# Fixed serving workloads against a running `pixi run serve-gemma4`.
# Usage: pixi run bash scripts/bench.sh <label> [scenario...]
# Results: results/bench/<label>/<scenario>.json (vLLM bench serve format).
set -euo pipefail

label=${1:?label}
shift
scenarios=("$@")
if [ ${#scenarios[@]} -eq 0 ]; then
  scenarios=(c1-128-128 c1-2048-128 c4-512-256 c1-8192-64)
fi
out="results/bench/$label"
mkdir -p "$out"

for s in "${scenarios[@]}"; do
  IFS=- read -r conc isl osl <<<"$s"
  conc=${conc#c}
  # One warm-up wave per scenario shape is excluded by running it separately first.
  prompts=$((conc * 4))
  for phase in warmup measure; do
    n=$([ "$phase" = warmup ] && echo "$conc" || echo "$prompts")
    args=(--backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions
      --model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized --served-model-name gemma4
      --dataset-name random --random-input-len "$isl" --random-output-len "$osl" --random-range-ratio 0
      --ignore-eos --seed 0 --num-prompts "$n" --max-concurrency "$conc"
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90)
    if [ "$phase" = measure ]; then
      args+=(--save-result --result-dir "$out" --result-filename "$s.json")
    fi
    vllm bench serve "${args[@]}" > "$out/$s.$phase.log" 2>&1
  done
  python - "$out/$s.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
keys = ["max_concurrency", "num_prompts", "median_ttft_ms", "median_tpot_ms", "output_throughput", "total_token_throughput", "median_e2el_ms"]
print(sys.argv[1].rsplit("/", 1)[-1], {k: round(r[k], 2) if isinstance(r.get(k), float) else r.get(k) for k in keys})
EOF
done
