"""Greedy outputs of concurrent requests must equal the same requests run alone.

    pixi run python scripts/vllm_consistency.py [--url http://127.0.0.1:8000]

Needs a running `pixi run serve-gemma4`. The concurrent requests stop at
different lengths, so decode steps through every active row count (and every
captured trace) while the others are mid-sequence.
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Name the three primary colors of paint.",
    "Write a short Python function that reverses a string.",
    "In one sentence, what did the Apollo 11 mission achieve?",
    "Write a haiku about autumn rain.",
]
LENGTHS = [64, 48, 32, 16]


def complete(url: str, prompt: str, max_tokens: int, **sampling) -> str:
    # The chat endpoint applies the template, which starts with <bos>; raw
    # completions get no <bos> from this tokenizer, and Gemma degrades without it.
    messages = [{"role": "user", "content": prompt}]
    sampling = sampling or {"temperature": 0}
    body = json.dumps({"model": "gemma4", "messages": messages, "max_tokens": max_tokens, **sampling}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    alone = [complete(args.url, p, n) for p, n in zip(PROMPTS, LENGTHS)]
    with ThreadPoolExecutor(len(PROMPTS)) as pool:
        together = list(pool.map(lambda pn: complete(args.url, *pn), zip(PROMPTS, LENGTHS)))
    # a second concurrent round after every batch size has replayed at least once
    with ThreadPoolExecutor(len(PROMPTS)) as pool:
        again = list(pool.map(lambda pn: complete(args.url, *pn), zip(PROMPTS, LENGTHS)))
    ok = True
    for i, (a, b, c) in enumerate(zip(alone, together, again)):
        same = a == b == c
        ok &= same
        print(f"[{'same' if same else 'DIFF'}] {PROMPTS[i]!r} -> {a[:70]!r}")
        if not same:
            print(f"    concurrent: {b[:70]!r}\n    again:      {c[:70]!r}")
    # Gemma's default sampling (top-k 64, top-p 0.95) goes through the model-side sampler.
    sampled = complete(args.url, PROMPTS[2], 48, temperature=1.0, top_k=64, top_p=0.95)
    print(f"[sampled] {sampled[:100]!r}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
