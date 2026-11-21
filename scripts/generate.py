"""Greedy generation outside vLLM, for checking the model end to end.

    pixi run python scripts/generate.py "In 1969, the Apollo 11 mission" --tokens 32 [--chat] [--trace]
"""

import argparse

import torch
from transformers import AutoTokenizer

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    ckpt = Checkpoint()
    tok = AutoTokenizer.from_pretrained(ckpt.root)
    if args.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"][0] if isinstance(ids, dict) or hasattr(ids, "keys") else ids[0]
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids[0]
    with open_n300(trace_region_size=200_000_000 if args.trace else 0) as mesh:
        model = Gemma4TT(mesh, ckpt, max_batch=1)
        pt = model.allocate_kv_cache(1, ids.shape[0] + args.tokens)
        out = [int(model.prefill(ids, 0, pt[:, 0]).argmax())]
        for i in range(args.tokens - 1):
            pos = ids.shape[0] + i
            logits = model.decode(torch.tensor([out[-1]]), torch.tensor([pos]), pt, trace=args.trace)
            out.append(int(logits[0].argmax()))
        model.release_traces()
    print("PROMPT", repr(tok.decode(ids)))
    print("OUTPUT", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
