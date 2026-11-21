"""Device time breakdown of traced decode steps from a tt-scope ops report.

    pixi run tt-scope report artifacts/runs/<run>
    pixi run python scripts/profile/trace_breakdown.py artifacts/runs/<run> \\
        [--phase decode1] [--layer-ops 73]

Under trace replay the CSV's OP TO OP LATENCY is host time between replays, so
gaps are recomputed from device firmware cycles: span = last FW end - first FW
start, gap = FW start of op i - FW end of op i-1 (dispatch, launch, and waits).
With --layer-ops, ops are folded by position inside a decoder layer, which
gives a per-op table averaged over layers.
"""

import argparse
from pathlib import Path

import pandas as pd

KERN = "DEVICE KERNEL DURATION [ns]"


def load(run: Path) -> pd.DataFrame:
    csv = run / "report/ops_perf_results.csv"
    cols = [
        "OP CODE",
        "OP TYPE",
        "DEVICE ID",
        KERN,
        "DEVICE FW START CYCLE",
        "DEVICE FW END CYCLE",
        "CORE COUNT",
        "INPUT_0_Y_PAD[LOGICAL]",
        "INPUT_0_X_PAD[LOGICAL]",
        "INPUT_1_Y_PAD[LOGICAL]",
        "INPUT_1_X_PAD[LOGICAL]",
    ]
    df = pd.read_csv(csv, usecols=cols.__contains__, low_memory=False)
    phase = "start"
    phases: list[str] = []
    for _, row in df.iterrows():
        if row["OP TYPE"] == "signpost":
            phase = row["OP CODE"]
        phases.append(phase)
    df["phase"] = phases
    return df[df["OP TYPE"] != "signpost"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--phase", default=None, help="default: every decode phase")
    ap.add_argument(
        "--layer-ops",
        type=int,
        default=0,
        help="ops per decoder layer, for the per-position table",
    )
    ap.add_argument(
        "--head-ops", type=int, default=1, help="ops before the first layer"
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=30,
        help="decoder layers (the rest is the LM head)",
    )
    ap.add_argument("--freq-mhz", type=float, default=1000.0)
    args = ap.parse_args()

    df = load(args.run)
    phases = (
        [args.phase]
        if args.phase
        else sorted(p for p in df["phase"].unique() if p.startswith("decode"))
    )
    for ph in phases:
        for dev, group in df[df["phase"] == ph].groupby("DEVICE ID"):
            g = group.sort_values("DEVICE FW START CYCLE")
            start, end = (
                g["DEVICE FW START CYCLE"].to_numpy(),
                g["DEVICE FW END CYCLE"].to_numpy(),
            )
            ns = 1e3 / args.freq_mhz
            span = (end.max() - start.min()) * ns / 1e6
            fw = ((end - start) * ns).sum() / 1e6
            kern = g[KERN].sum() / 1e6
            gaps = ((start[1:] - end[:-1]).clip(min=0) * ns).sum() / 1e6
            print(
                f"{ph} chip {dev}: {len(g)} ops, span {span:.2f} ms, FW {fw:.2f} ms, "
                f"kernel {kern:.2f} ms, gaps {gaps:.2f} ms "
                f"({gaps / len(g) * 1e3:.2f} us/op)"
            )
    if not args.layer_ops:
        return
    ph = phases[min(1, len(phases) - 1)]
    g = (
        df[(df["phase"] == ph) & (df["DEVICE ID"] == df["DEVICE ID"].min())]
        .sort_values("DEVICE FW START CYCLE")
        .reset_index(drop=True)
    )
    start, end = g["DEVICE FW START CYCLE"], g["DEVICE FW END CYCLE"]
    g["gap_us"] = (start - end.shift(1)).clip(lower=0).fillna(0) / args.freq_mhz
    g["kern_us"] = g[KERN] / 1e3
    body = g.iloc[args.head_ops :]
    n_layers = min(args.layers, len(body) // args.layer_ops)
    layers = body.iloc[: n_layers * args.layer_ops].copy()
    layers["pos"] = [i % args.layer_ops for i in range(len(layers))]
    tab = layers.groupby("pos").agg(
        op=("OP CODE", "first"),
        a=("INPUT_0_X_PAD[LOGICAL]", "first"),
        b=("INPUT_1_X_PAD[LOGICAL]", "first"),
        cores=("CORE COUNT", "max"),
        kern_us=("kern_us", "mean"),
        kern_max=("kern_us", "max"),
        gap_us=("gap_us", "mean"),
    )
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nper position, {ph} chip 0, mean over {n_layers} layers:")
    print(tab.round(1).to_string())
    tail = g.iloc[args.head_ops + n_layers * args.layer_ops :]
    print(
        f"\nlayer total kernel {tab.kern_us.sum():.0f} us, gaps {tab.gap_us.sum():.0f} "
        "us; "
        f"tail {len(tail)} ops kernel {tail.kern_us.sum():.0f} us, gaps "
        f"{tail.gap_us.sum():.0f} us"
    )


if __name__ == "__main__":
    main()
