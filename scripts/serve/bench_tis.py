"""Serving points built like tt-inference-server's release client.

    pixi run python scripts/serve/bench_tis.py <label> [point ...]

Runs against a running `just serve-gemma4`, so the results can sit next to the
official gates. A point is c<concurrency>-<isl>-<osl>-n<prompts>[-sampled]; the
default is the official c1 gates that fit a 32K server, plus our c4 point.

Matches the release client (vLLM 0.13 `vllm bench serve`): openai-chat backend
on /v1/chat/completions, random dataset, fixed ISL/OSL, ignore-eos, temperature
0 (0.13 hard-codes it; our client does not, so it is passed), and tput_user =
1000 / mean TPOT. Each point gets one discarded warm-up pass and three measured
repetitions with seeds 0..2. "-sampled" drops --temperature, so the server's
generation-config sampling (temperature 1, top_k 64) applies.

Results: artifacts/serve/<label>/<point>.{warmup,r0,r1,r2}.{json,log}
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_POINTS = ("c1-128-128-n8", "c1-2048-128-n4", "c1-8192-128-n2", "c4-128-128-n16")
# The warm-up pass uses its own seed so it never repeats a measured one.
SEEDS = {"warmup": 100, "r0": 0, "r1": 1, "r2": 2}


@dataclass(frozen=True)
class Point:
    """One serving point."""

    name: str
    concurrency: int
    isl: int
    osl: int
    prompts: int
    sampled: bool


def parse_point(name: str) -> Point:
    """Parse c<concurrency>-<isl>-<osl>-n<prompts>[-sampled]."""
    conc, isl, osl, prompts, *mode = name.split("-")
    if (
        mode not in ([], ["sampled"])
        or not conc.startswith("c")
        or not prompts.startswith("n")
    ):
        msg = f"bad point {name!r}: expected c<conc>-<isl>-<osl>-n<prompts>[-sampled]"
        raise ValueError(msg)
    return Point(
        name, int(conc[1:]), int(isl), int(osl), int(prompts[1:]), sampled=bool(mode)
    )


def bench_args(point: Point, seed: int, out: Path, filename: str) -> list[str]:
    """Build the `vllm bench serve` command for one pass."""
    args = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", "http://127.0.0.1:8000",
        "--endpoint", "/v1/chat/completions",
        "--model", "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "--served-model-name", "gemma4",
        "--dataset-name", "random",
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--random-range-ratio", "0",
        "--ignore-eos",
        "--seed", str(seed),
        "--num-prompts", str(point.prompts),
        "--max-concurrency", str(point.concurrency),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result", "--save-detailed",
        "--result-dir", str(out),
        "--result-filename", filename,
    ]  # fmt: skip
    if not point.sampled:
        args += ["--temperature", "0"]
    return args


def summarize(result: dict[str, float]) -> dict[str, float]:
    """Pick the numbers that the official gates report."""
    return {
        "in_tok": round(result["total_input_tokens"] / result["num_prompts"], 1),
        "mean_ttft": round(result["mean_ttft_ms"], 1),
        "mean_tpot": round(result["mean_tpot_ms"], 2),
        "tput_user": round(1000 / result["mean_tpot_ms"], 2),
        "tput": round(result["output_throughput"], 2),
    }


def main() -> None:
    """Run every pass of every point, then print its summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label")
    parser.add_argument("points", nargs="*", default=DEFAULT_POINTS)
    args = parser.parse_args()
    out = Path("artifacts/serve") / args.label
    out.mkdir(parents=True, exist_ok=True)
    for point in map(parse_point, args.points):
        for rep, seed in SEEDS.items():
            stem = f"{point.name}.{rep}"
            with (out / f"{stem}.log").open("w") as log:
                subprocess.run(  # noqa: S603 - fixed program, arguments built here
                    bench_args(point, seed, out, f"{stem}.json"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = json.loads((out / f"{stem}.json").read_text())
            print(point.name, rep, summarize(result))


if __name__ == "__main__":
    main()
