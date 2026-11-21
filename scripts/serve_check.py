"""Request lifecycle checks against a running server: outputs must not depend on
what else the server is doing.

    pixi run python scripts/serve_check.py [--url http://127.0.0.1:8000] [check ...]

Needs a running `pixi run serve-gemma4`. Every check but tools compares greedy token ids
with the same request run alone:

- isolation: four chat requests at once, stopping at different lengths, so decode
  replays every active row count while the others are mid-sequence. One row
  runs the fused batch-1 kernels and 2-4 rows the batched ones, which round
  differently, and which one computed a step depends on arrival timing; so a
  request may take the other branch at a near-tie later on. Cross-request
  contamination garbles a request at once, so this check needs the first 16
  tokens to agree and reports later divergences;
- raw: a raw token prompt (no <bos>, flat logits, the investigation's failing
  case), with temperature 0 set explicitly;
- arrival: an 8K-token prompt arrives while a stream decodes; reports the
  stream's longest gap between tokens;
- cancel: a stream closed after its first token, then a request in its place;
- prefix: a second chat turn that extends the first, with the prefix cache and
  with a cache_salt that misses it; reports TTFT and the agreeing tokens;
- seeded: a sampled request with a seed, twice;
- preempt (not in the default set): four requests that outgrow a small KV pool
  together, so vLLM preempts and resumes some; needs the server started with a
  small pool, see check_preempt;
- tools: tool_choice auto, required and named, with and without thinking: a
  required or named choice must return a call (the named one) even for a
  prompt that needs no tool, and auto must still answer such a prompt in text.

The prefix check compares a cache hit with a fresh prefill: the hit reads BFP8
K/V from the cache where the fresh prefill used BF16 rows, so a near-tie can
flip a late token. It fails only if the first 16 tokens differ.
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tt_models.models.gemma4.checkpoint import resolve_snapshot

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


def metric(url: str, name: str) -> float:
    """A vLLM counter from /metrics, summed over its labels."""
    text = urllib.request.urlopen(f"{url}/metrics", timeout=30).read().decode()
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name + "{") or line.startswith(name + " "))


def prefix_hits(url: str) -> float:
    """Prompt tokens served from the prefix cache so far."""
    return metric(url, "vllm:prefix_cache_hits_total")


def stream(url: str, prompt: list[int], max_tokens: int, stop_after: int | None = None, first=None) -> tuple[list[int], list[float]]:
    """Stream a raw-token greedy completion; returns its token ids and arrival times.
    stop_after closes the connection after that many tokens, which aborts the request.
    first, a threading.Event, is set when the first token arrives."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True, "return_token_ids": True, **GREEDY}
    resp = post(url, "/v1/completions", body, stream=True)
    ids, times = [], []
    with resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            new = choice.get("token_ids") or []
            ids += new
            times.append(time.perf_counter())
            if first is not None and new:
                first.set()
            if stop_after is not None and len(ids) >= stop_after:
                break
    return ids, times


def compare(name: str, ref: list[int], got: list[int], need: int | None = None) -> bool:
    agree = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
    ok = agree >= (need if need is not None else max(len(ref), len(got)))
    label = "DIFF" if not ok else "same" if agree == max(len(ref), len(got)) else "near-tie"
    print(f"[{label}] {name}: {agree}/{len(ref)} tokens agree", flush=True)
    return ok


def tokens(url: str, text: str, bos: bool) -> list[int]:
    ids = post(url, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
    return ([2] if bos else []) + ids  # 2 is Gemma's <bos>


def document(n_chars: int) -> str:
    """The checkpoint's model card, repeated to n_chars: a long prompt that stays the same as this
    repository changes (prompts built from the repository's own files moved its near-ties)."""
    text = (resolve_snapshot() / "README.md").read_text()
    return (text * (n_chars // len(text) + 1))[:n_chars]


def check_isolation(url):
    alone = [chat(url, p, n) for p, n in zip(PROMPTS, LENGTHS)]
    ok = True
    for rnd in range(2):  # the second round replays every captured trace again
        with ThreadPoolExecutor(len(PROMPTS)) as pool:
            together = list(pool.map(lambda pn: chat(url, *pn), zip(PROMPTS, LENGTHS)))
        for p, a, b in zip(PROMPTS, alone, together):
            ok &= compare(f"concurrent round {rnd}: {p[:40]!r}", a, b, need=min(16, len(a)))
    return ok


def check_raw(url):
    prompt = list(range(1000, 1128))  # no <bos>: flat logits, where any difference shows
    a, b = complete(url, prompt, 96), complete(url, prompt, 96)
    return compare("raw tokens, temperature 0, twice", a, b)


def check_arrival(url):
    short = tokens(url, "Count from one to fifty in English words, separated by commas:", bos=True)
    ref = complete(url, short, 128)
    long = tokens(url, document(40000), bos=True)[:8192]
    decoding = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        streamed = pool.submit(stream, url, short, 128, first=decoding)
        if not decoding.wait(timeout=120):
            streamed.result(timeout=1)  # surfaces the stream's error, if it failed
            raise TimeoutError("the stream produced no token within 120 s")
        t0 = time.perf_counter()
        # A fresh cache_salt keeps an earlier run's blocks from serving this prompt.
        arrived = pool.submit(lambda: (complete(url, long, 1, cache_salt=f"arrival-{random.random()}"), time.perf_counter() - t0)[1])
        ids, times = streamed.result()
        long_ttft = arrived.result()
    gaps = [b - a for a, b in zip(times, times[1:])]
    print(
        f"    stream gaps: median {sorted(gaps)[len(gaps) // 2] * 1e3:.1f} ms, max {max(gaps) * 1e3:.0f} ms;"
        f" the 8192-token request's first token after {long_ttft:.2f} s",
        flush=True,
    )
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
    h0 = prefix_hits(url)
    t0 = time.perf_counter()
    hit = chat(url, turn2, 48)
    t_hit = time.perf_counter() - t0
    h1 = prefix_hits(url)
    t0 = time.perf_counter()
    fresh = chat(url, turn2, 48, cache_salt=f"miss-{random.random()}")
    t_fresh = time.perf_counter() - t0
    h2 = prefix_hits(url)
    print(
        f"    turn 2 from the cache: {h1 - h0:.0f} prompt tokens hit, {t_hit:.2f} s; fresh: {h2 - h1:.0f} hit, {t_fresh:.2f} s (48 tokens each)",
        flush=True,
    )
    ok = h1 - h0 > 0 and h2 == h1
    if not ok:
        print("[DIFF] the cached turn missed the prefix cache, or the salted turn hit it", flush=True)
    return compare("turn 2 from the prefix cache vs fresh", fresh, hit, need=16) and ok


def check_seeded(url):
    sampled = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1234}
    a = chat(url, PROMPTS[2], 48, **sampled)
    b = chat(url, PROMPTS[2], 48, **sampled)
    return compare("seeded sampling, twice", a, b)


def check_preempt(url):
    """Start the server with a pool too small for four of these requests:

        GEMMA4_KV_POOL_GIB=0.2 pixi run serve-gemma4 --max-model-len 4096

    vLLM admits a request only when its prompt fits, so the four short prompts
    start together and run out of KV pages as they grow: at 1,560 tokens each
    needs about 98 pages (13 global, 17 per sliding group), against the 313 that
    0.2 GiB holds. vLLM then preempts the newest request and later recomputes it
    from its prompt and the tokens it had. Listing numbers has a clear next token
    at every step, so batch numerics do not flip it and the outputs must agree."""
    starts = [1, 5000, 20000, 70000]
    prompts = [f"Write every integer from {n} to {n + 700} on its own line. Do not write code; output only the numbers." for n in starts]
    salts = [f"preempt-{random.random()}" for _ in prompts]  # no prefix hits between the runs
    run = {"chat_template_kwargs": {"enable_thinking": False}}
    alone = [chat(url, p, 1536, cache_salt=s, **run) for p, s in zip(prompts, salts)]
    short = [n for n, a in zip(starts, alone) if len(a) < 1536]
    if short:  # each request must outlive the page shortage to cover the resume
        print(f"[DIFF] the lists from {short} ended early; pick other start values", flush=True)
    before = metric(url, "vllm:num_preemptions_total")
    with ThreadPoolExecutor(len(prompts)) as pool:
        together = list(pool.map(lambda ps: chat(url, ps[0], 1536, cache_salt=ps[1] + "-b", **run), zip(prompts, salts)))
    preempted = metric(url, "vllm:num_preemptions_total") - before
    print(f"    {preempted:.0f} preemptions during the concurrent run", flush=True)
    ok = preempted > 0 and not short
    if not preempted:
        print("[DIFF] no request was preempted; start the server with a smaller pool", flush=True)
    for n, a, b in zip(starts, alone, together):
        ok &= compare(f"numbers from {n} with preemption vs alone", a, b)
    return ok


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
CLOCK = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def check_tools(url):
    """Tool choice is enforced by the structural tag of gemma4_tt/tool_parser.py."""
    tools = [WEATHER, CLOCK]
    named = {"type": "function", "function": {"name": "get_time"}}
    cases = [  # (prompt, tool_choice, expected call or None for a text answer)
        ("What's the weather in Taipei?", "auto", "get_weather"),
        ("Hi there, how are you?", "auto", None),
        ("Hi there, how are you?", "required", {"get_weather", "get_time"}),
        ("What's the weather in Taipei?", named, "get_time"),
    ]
    ok = True
    for thinking in (False, True):
        for prompt, choice, want in cases:
            body = {
                "model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, **GREEDY,
                "tools": tools, "tool_choice": choice, "chat_template_kwargs": {"enable_thinking": thinking},
            }
            choice0 = post(url, "/v1/chat/completions", body)["choices"][0]
            calls = choice0["message"].get("tool_calls") or []
            names = [c["function"]["name"] for c in calls]
            args_ok = all(isinstance(json.loads(c["function"]["arguments"]), dict) for c in calls)
            if want is None:
                good = not calls and bool(choice0["message"].get("content"))
            else:
                if isinstance(choice, str):
                    good = bool(names) and set(names) <= ({want} if isinstance(want, str) else want)
                else:  # a named choice allows exactly one call, to that tool
                    good = names == [want]
                good &= args_ok
                # OpenAI (and vLLM) finish a forced named call with "stop"
                good &= choice0["finish_reason"] in (("tool_calls",) if isinstance(choice, str) else ("stop", "tool_calls"))
            label = choice if isinstance(choice, str) else "named get_time"
            print(f"[{'OK' if good else 'DIFF'}] {label}, thinking {'on' if thinking else 'off'}: "
                  f"{names or repr((choice0['message'].get('content') or '')[:60])} ({choice0['finish_reason']})", flush=True)
            ok &= good
    return ok


CHECKS = {
    "isolation": check_isolation,
    "raw": check_raw,
    "arrival": check_arrival,
    "cancel": check_cancel,
    "prefix": check_prefix,
    "seeded": check_seeded,
    "tools": check_tools,
}
EXTRA_CHECKS = {"preempt": check_preempt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("checks", nargs="*", help=f"any of {', '.join(CHECKS | EXTRA_CHECKS)} (default: {', '.join(CHECKS)})")
    args = ap.parse_args()
    known = CHECKS | EXTRA_CHECKS
    unknown = set(args.checks) - set(known)
    if unknown:
        ap.error(f"unknown checks: {', '.join(sorted(unknown))}")
    failed = [name for name in args.checks or CHECKS if not known[name](args.url)]
    print("PASS" if not failed else f"FAIL: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
