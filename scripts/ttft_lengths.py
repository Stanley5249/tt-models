"""Latency of one-token chat requests at prompt lengths the server has not seen.

    pixi run python scripts/ttft_lengths.py [seed]

Needs a running `pixi run serve-gemma4`. Sends eight prompts of 400-3,700 tokens
(repository source, lengths drawn from the seed) one at a time, each with a fresh
cache_salt and one output token, and prints the wall time of each request: the
time to first token plus the response's finalization and transfer. A length
whose prefill programs were never built pays for building them here.
"""

import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_check import MODEL, document, post  # noqa: E402

URL = "http://127.0.0.1:8000"


def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    doc = document(40000)
    times = []
    for chars in (1200, 1900, 2600, 4100, 5300, 7000, 9100, 11500):
        chars += rng.randint(0, 300)
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": doc[:chars] + "\nSummarize."}],
            "max_tokens": 1,
            "temperature": 0,
            "cache_salt": f"len-{uuid.uuid4()}",
        }
        t0 = time.perf_counter()
        usage = post(URL, "/v1/chat/completions", body)["usage"]
        times.append(time.perf_counter() - t0)
        print(f"{usage['prompt_tokens']} prompt tokens: {times[-1]:.2f} s", flush=True)
    print(f"mean {sum(times) / len(times):.2f} s")


if __name__ == "__main__":
    main()
