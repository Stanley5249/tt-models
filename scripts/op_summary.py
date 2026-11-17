"""Summarize a tt-scope ops_perf_results CSV by signpost phase and op.

    pixi run python scripts/op_summary.py artifacts/runs/<run> --phase decode1 [--by shape]

Per chip: total device kernel time, op-to-op gap time, and the top ops by
kernel time. Only the small per-op CSV is read, never the raw device log.
"""

import argparse
from pathlib import Path

import pandas as pd

COLS = [
    "OP CODE",
    "OP TYPE",
    "DEVICE ID",
    "DEVICE KERNEL DURATION [ns]",
    "OP TO OP LATENCY [ns]",
    "CORE COUNT",
    "MATH FIDELITY",
    "INPUT_0_Y_PAD[LOGICAL]",
    "INPUT_0_X_PAD[LOGICAL]",
    "INPUT_1_Y_PAD[LOGICAL]",
    "INPUT_1_X_PAD[LOGICAL]",
    "INPUT_0_DATATYPE",
    "INPUT_1_DATATYPE",
    "INPUT_0_MEMORY",
]


def load(run: Path) -> pd.DataFrame:
    csv = next(run.glob("reports/*/ops_perf_results_*.csv"))
    df = pd.read_csv(csv, usecols=lambda c: c in COLS, low_memory=False)
    phase, phases = "start", []
    for _, row in df.iterrows():
        if row["OP TYPE"] == "signpost":
            phase = row["OP CODE"]
        phases.append(phase)
    df["phase"] = phases
    return df[df["OP TYPE"] != "signpost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--by", choices=["op", "shape"], default="op")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    df = load(args.run)
    print("phases:", df.groupby("phase").size().to_dict())
    if args.phase:
        df = df[df["phase"] == args.phase]
    kern = "DEVICE KERNEL DURATION [ns]"
    gap = "OP TO OP LATENCY [ns]"
    if args.by == "shape":
        df["key"] = (
            df["OP CODE"]
            + " a=" + df["INPUT_0_Y_PAD[LOGICAL]"].astype(str) + "x" + df["INPUT_0_X_PAD[LOGICAL]"].astype(str)
            + " b=" + df["INPUT_1_Y_PAD[LOGICAL]"].astype(str) + "x" + df["INPUT_1_X_PAD[LOGICAL]"].astype(str)
            + " " + df["INPUT_1_DATATYPE"].astype(str)
        )
    else:
        df["key"] = df["OP CODE"]
    for dev, g in df.groupby("DEVICE ID"):
        total, gaps = g[kern].sum() / 1e6, g[gap].sum() / 1e6
        print(f"\n== chip {dev}: {len(g)} ops, kernel {total:.2f} ms, op-to-op gaps {gaps:.2f} ms")
        top = g.groupby("key").agg(count=(kern, "size"), ms=(kern, lambda s: s.sum() / 1e6), avg_us=(kern, lambda s: s.mean() / 1e3))
        top = top.sort_values("ms", ascending=False).head(args.top)
        top["share"] = (top["ms"] / total * 100).round(1)
        print(top.round(3).to_string())


if __name__ == "__main__":
    main()
