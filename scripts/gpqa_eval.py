"""GPQA Diamond accuracy of a running server, as upstream's QB2 release check scores it.

    uv run --no-project --with-requirements scripts/gpqa_eval.requirements.txt \\
        python scripts/gpqa_eval.py --count 10 --out results/quality/gpqa-<label>

Needs a running `pixi run serve-gemma4` and Hugging Face access to the gated
Idavidrein/gpqa dataset. Port of tt-metal's
models/demos/gemma4_26b_a4b_qb2/tests/benchmark.py (Apache-2.0): the same
lm-eval task (r1_gpqa_diamond at the pinned dataset revision, choices shuffled
with seed 42), sampling (temperature 1, top-p 0.95, top-k 20, seed 42, thinking
off, 32768 tokens) and exact-match scoring. Upstream gates on the first 10
questions at 0.8; --count takes more of the 198 for a tighter estimate.
"""

import argparse
import json
import random
import time
from pathlib import Path

import httpx

DATASET_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
MAX_TOKENS = 32768
SEED = 42


def load_gpqa(count: int):
    import datasets
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config

    # Choice permutations must come from this seed, not a previous map cache.
    datasets.disable_caching()
    random.seed(SEED)
    config = load_yaml_config(TaskManager().task_index["r1_gpqa_diamond"]["yaml_path"])
    config["dataset_kwargs"] = {"revision": DATASET_REVISION}
    task = ConfigurableTask(config=config)
    docs = task.validation_docs()
    if len(docs) != 198:
        raise ValueError(f"expected 198 GPQA Diamond questions, got {len(docs)}")
    return task, [(i, docs[i], task.doc_to_text(docs[i])) for i in range(count)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").unlink(missing_ok=True)  # a failed rerun must not leave the old summary
    task, cases = load_gpqa(args.count)
    rows = []
    with httpx.Client(base_url=args.url, timeout=3600) as client, (args.out / "responses.jsonl").open("w") as log:
        for i, doc, prompt in cases:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "seed": SEED,
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": [],
            }
            t0 = time.perf_counter()
            response = client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            text = choice["message"].get("content") or ""
            correct = int(task.process_results(doc, [text])["exact_match"])
            row = {
                "id": i,
                "correct": correct,
                "finish_reason": choice["finish_reason"],
                "completion_tokens": response.json()["usage"]["completion_tokens"],
                "seconds": round(time.perf_counter() - t0, 1),
                "text": text,
            }
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"GPQA {i}: correct={correct} tokens={row['completion_tokens']} {row['seconds']} s", flush=True)
    correct = sum(r["correct"] for r in rows)
    summary = {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "first10_correct": sum(r["correct"] for r in rows[:10]),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
