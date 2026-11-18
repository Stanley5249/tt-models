#!/usr/bin/env bash
# Standalone decode for one checkout: blocking trace time (b1) and end-to-end b1/b4, 3 runs each.
# Usage: pixi run bash scripts/final_standalone.sh <checkout dir> <label>   (run from tt-vllm)
set -uo pipefail
dir=$1; label=$2
out=$PWD/results/final/standalone-$label.txt
cfg="governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), profile $(powerprofilesctl get 2>/dev/null)"
echo "$label $(git -C "$dir" rev-parse --short HEAD) ttnn $(python -c 'import importlib.metadata as m; print(m.version("ttnn"))' 2>/dev/null) host: $cfg" > "$out"
cd "$dir"
export PYTHONPATH=$PWD
for i in 1 2 3; do python "$READ_TIME" 2>&1 | grep -oE "execute [0-9.]+" >> "$out"; done
for b in 1 4; do for i in 1 2 3; do
  python scripts/bench_decode.py --batch $b --steps 128 --trace 2>&1 | grep -oE '"median_ms": [0-9.]+, "mean_ms": [0-9.]+' | sed "s/^/b$b /" >> "$out"
done; done
cat "$out"
