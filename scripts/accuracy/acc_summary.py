"""Summarize compare.py JSON results: top-1 and mean KL over the decode rows.

pixi run python scripts/accuracy/acc_summary.py artifacts/accuracy/*.json
"""

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    with Path(path).open() as file:
        rows = [r for r in json.load(file)["rows"] if "sec" in r]
    top1 = sum(r["top1"] for r in rows) / len(rows)
    kl = sum(r["kl"] for r in rows) / len(rows)
    top5 = sum(r["top5_overlap"] for r in rows) / len(rows)
    print(
        f"{path}: rows={len(rows)} top1={top1:.3f} top5_overlap={top5:.2f} kl={kl:.4f}"
    )
