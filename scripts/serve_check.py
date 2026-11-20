"""Request lifecycle checks against a running server: outputs must not depend on
what else the server is doing.

    pixi run python scripts/serve_check.py [--url http://127.0.0.1:8000] [check ...]

Needs a running `pixi run serve-gemma4`. Every check compares greedy token ids
with the same request run alone:

- isolation: four chat requests at once, stopping at different lengths, so decode
  replays every active row count while the others are mid-sequence;
- raw: a raw token prompt (no <bos>, flat logits, the investigation's failing
  case), with temperature 0 set explicitly;
- arrival: an 8K-token prompt arrives while a stream decodes; reports the
  stream's longest gap between tokens;
- cancel: a stream closed after its first token, then a request in its place;
- prefix: a second chat turn that extends the first, with the prefix cache and
  with a cache_salt that misses it; reports TTFT and the agreeing tokens;
- seeded: a sampled request with a seed, twice.

The prefix check compares a cache hit with a fresh prefill: the hit reads BFP8
K/V from the cache where the fresh prefill used BF16 rows, so a near-tie can
flip a late token. It fails only if the first 16 tokens differ.
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "gemma4"
PROMPTS = [
    "Name the three primary colors of paint.",
    "Write a short Python function that reverses a string.",
    "In one sentence, what did the Apollo 11 mission achieve?",
    "Write a haiku about autumn rain.",
]
LENGTHS = [64, 48, 32, 16]
GREEDY = {"temperature": 0}


def post(url: str, path: str, body: dict, stream: bool = False):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=900)
    return resp if stream else json.load(resp)


def chat(url: str, prompt: str | list[dict], max_tokens: int, **extra) -> list[int]:
    """Token ids of a chat completion; thinking tokens count, since they are sampled too."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "return_token_ids": True, **GREEDY, **extra}
    return post(url, "/v1/chat/completions", body)["choices"][0]["token_ids"]


def complete(url: str, prompt: list[int], max_tokens: int, **extra) -> list[int]:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "return_token_ids": True, **GREEDY, **extra}
    return post(url, "/v1/completions", body)["choices"][0]["token_ids"]


def stream(url: str, prompt: list[int], max_tokens: int, stop_after: int | None = None) -> tuple[list[int], list[float]]:
    """Stream a raw-token greedy completion; returns its token ids and arrival times.
    stop_after closes the connection after that many tokens, which aborts the request."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True, "return_token_ids": True, **GREEDY}
    resp = post(url, "/v1/completions", body, stream=True)
    ids, times = [], []
    with resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            ids += choice.get("token_ids") or []
            times.append(time.perf_counter())
            if stop_after is not None and len(ids) >= stop_after:
                break
    return ids, times


def compare(name: str, ref: list[int], got: list[int], need: int | None = None) -> bool:
    agree = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
    ok = agree >= (need if need is not None else max(len(ref), len(got)))
    print(f"[{'same' if ok else 'DIFF'}] {name}: {agree}/{len(ref)} tokens agree", flush=True)
    return ok


def tokens(url: str, text: str, bos: bool) -> list[int]:
    ids = post(url, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
    return ([2] if bos else []) + ids  # 2 is Gemma's <bos>


def document(n_chars: int) -> str:
    """Repository source as a long, realistic prompt (an agent reading files)."""
    root = Path(__file__).resolve().parents[1]
    text = "".join(p.read_text() for p in sorted((root / "gemma4_tt").rglob("*.py")))
    return (text * (n_chars // len(text) + 1))[:n_chars]


def check_isolation(url):
    alone = [chat(url, p, n) for p, n in zip(PROMPTS, LENGTHS)]
    ok = True
    for rnd in range(2):  # the second round replays every captured trace again
        with ThreadPoolExecutor(len(PROMPTS)) as pool:
            together = list(pool.map(lambda pn: chat(url, *pn), zip(PROMPTS, LENGTHS)))
        for p, a, b in zip(PROMPTS, alone, together):
            ok &= compare(f"concurrent round {rnd}: {p[:40]!r}", a, b)
    return ok


def check_raw(url):
    prompt = list(range(1000, 1128))  # no <bos>: flat logits, where any difference shows
    a, b = complete(url, prompt, 96), complete(url, prompt, 96)
    return compare("raw tokens, temperature 0, twice", a, b)


def check_arrival(url):
    short = tokens(url, "Count from one to fifty in English words, separated by commas:", bos=True)
    ref = complete(url, short, 128)
    long = tokens(url, document(40000), bos=True)[:8192]
    with ThreadPoolExecutor(2) as pool:
        streamed = pool.submit(stream, url, short, 128)
        time.sleep(0.5)  # the stream is decoding when the long prompt arrives
        arrived = pool.submit(complete, url, long, 16)
        ids, times = streamed.result()
        arrived.result()
    gaps = [b - a for a, b in zip(times, times[1:])]
    print(f"    stream gaps: median {sorted(gaps)[len(gaps) // 2] * 1e3:.1f} ms, max {max(gaps) * 1e3:.0f} ms", flush=True)
    return compare("stream while an 8192-token prompt arrives", ref, ids)


def check_cancel(url):
    prompt = tokens(url, "Write a long story about a lighthouse keeper.", bos=True)
    ref = complete(url, prompt, 64)
    stream(url, prompt, 512, stop_after=1)
    time.sleep(0.5)
    return compare("after a cancelled stream", ref, complete(url, prompt, 64))


def check_prefix(url):
    doc = document(24000)
    turn1 = [{"role": "user", "content": f"Here is some code:\n\n{doc}\n\nSummarize what the first class does."}]
    answer = post(url, "/v1/chat/completions", {"model": MODEL, "messages": turn1, "max_tokens": 64, **GREEDY})["choices"][0]["message"]
    turn2 = turn1 + [
        {"role": "assistant", "content": answer.get("content") or ""},
        {"role": "user", "content": "Now name one function that touches the KV cache."},
    ]
    t0 = time.perf_counter()
    hit = chat(url, turn2, 48)
    t_hit = time.perf_counter() - t0
    t0 = time.perf_counter()
    fresh = chat(url, turn2, 48, cache_salt=f"miss-{random.random()}")
    t_fresh = time.perf_counter() - t0
    print(f"    turn 2 with the prefix cache {t_hit:.2f} s, without {t_fresh:.2f} s (48 tokens each)", flush=True)
    return compare("turn 2 from the prefix cache vs fresh", fresh, hit, need=16)


def check_seeded(url):
    sampled = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}
    a = chat(url, PROMPTS[2], 48, **sampled)
    b = chat(url, PROMPTS[2], 48, **sampled)
    return compare("seeded sampling, twice", a, b)


CHECKS = {
    "isolation": check_isolation,
    "raw": check_raw,
    "arrival": check_arrival,
    "cancel": check_cancel,
    "prefix": check_prefix,
    "seeded": check_seeded,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("checks", nargs="*", help=f"any of {', '.join(CHECKS)} (default: all)")
    args = ap.parse_args()
    unknown = set(args.checks) - set(CHECKS)
    if unknown:
        ap.error(f"unknown checks: {', '.join(sorted(unknown))}")
    failed = [name for name in args.checks or CHECKS if not CHECKS[name](args.url)]
    print("PASS" if not failed else f"FAIL: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
