"""Greedy outputs of a few chat prompts, saved or compared against a saved run.

    pixi run python scripts/accuracy/greedy_check.py --save scripts/accuracy/greedy-base.json
    pixi run python scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json

Uses traced decode, as serving does. A comparison reports, per prompt, how
many leading tokens match up to the reference's first end-of-turn token
(serving stops there, and the text after it is flat and near-tied).
Numerically different kernels can legitimately diverge after a near-tie, so
read the texts when a prompt diverges early.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT

PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain why the sky is blue to a ten-year-old.",
    "List three prime numbers greater than 100.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--precision", default="")
    args = ap.parse_args()
    ckpt = Checkpoint()
    tok = AutoTokenizer.from_pretrained(ckpt.root)
    results = []
    with open_n300(trace_region_size=300_000_000) as mesh:
        model = Gemma4TT(mesh, ckpt, max_batch=1)
        pt = model.allocate_kv_cache(1, 4096)
        for prompt in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"][0] if hasattr(ids, "keys") else ids[0]
            logits = model.prefill(ids, 0, pt[:, 0])
            out = [int(logits.argmax())]
            for step in range(args.tokens - 1):
                pos = torch.tensor([len(ids) + step])
                logits = model.decode(torch.tensor([out[-1]]), pos, pt, trace=True)
                out.append(int(logits[0].argmax()))
            results.append({"prompt": prompt, "ids": out, "text": tok.decode(out)})
        model.release_traces()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=1))
    ref = json.loads(args.compare.read_text()) if args.compare else None
    eos = set(tok.convert_tokens_to_ids(["<turn|>"])) | {tok.eos_token_id}
    for i, r in enumerate(results):
        line = repr(r["text"][:100])
        if ref:
            b = ref[i]["ids"]
            b = b[: next((j + 1 for j, t in enumerate(b) if t in eos), len(b))]
            a = r["ids"][: len(b)]
            same = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            line = f"match {same}/{len(a)}  " + line
            if same < len(a):
                line += f"\n    ref: {ref[i]['text'][:100]!r}"
        print(line)


if __name__ == "__main__":
    main()
