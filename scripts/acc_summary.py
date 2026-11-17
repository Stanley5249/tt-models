"""Summarize compare.py JSON results: top-1 and mean KL over the decode rows.

    pixi run python scripts/acc_summary.py results/accuracy/decode128-*.json
"""

import json
import sys

for path in sys.argv[1:]:
    rows = json.load(open(path))["rows"][1:]  # row 0 is the prefill logit
    top1 = sum(r["top1"] for r in rows) / len(rows)
    kl = sum(r["kl"] for r in rows) / len(rows)
    top5 = sum(r["top5_overlap"] for r in rows) / len(rows)
    print(f"{path}: rows={len(rows)} top1={top1:.3f} top5_overlap={top5:.2f} kl={kl:.4f}")
