# tt-models

TTNN model implementations served through vLLM on Tenstorrent hardware. The
one model today is Gemma 4 26B-A4B (the QAT checkpoint) on a Wormhole N300,
tensor-parallel over its two chips, with custom generic-op kernels where
ttnn ops leave DRAM bandwidth unused.

## Setup

The environment is a [pixi](https://pixi.sh) workspace: Python 3.12, a
Tracy-enabled ttnn wheel, CPU torch, vLLM and vllm-tt-plugin, and this
package installed editable.

    pixi install
    pixi run serve-gemma4          # OpenAI-compatible server on 127.0.0.1:8000
    pixi run serve-qwen3-8b        # tt_transformers' Qwen3-8B

Everything reads the checkpoint from the Hugging Face cache and never
downloads it, so fetch it once with
`pixi run hf download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`. The
device tests use random weights but still read its `config.json`.

The first start converts the checkpoint into a ttnn weight cache under
`cache/` and builds the prefill programs, which takes a while; later starts
load the cache.

## Layout

| Path | What |
|---|---|
| `src/tt_models/common/` | Model-independent helpers: mesh, compute configs, weight cache, collectives, norms, trace timing |
| `src/tt_models/models/gemma4/` | Gemma 4: model, layers, MoE, KV cache, vLLM adapter, tool parser, HF reference |
| `src/tt_models/models/gemma4/kernels/` | Reader, writer and compute kernels of the generic ops |
| `scripts/serve/` | Checks and benchmarks against a running server |
| `scripts/accuracy/` | Teacher-forced accuracy against the fp32 reference, greedy checks, GPQA |
| `scripts/bench/` | Standalone decode and prefill timing, soak test |
| `scripts/kernels/` | Device checks of single kernels against torch or ttnn |
| `scripts/profile/` | tt-scope workloads and trace breakdowns |
| `tests/` | pytest: host tests by default, `-m device` for the N300 |
| `docs/serving.md` | How the backend works, what it supports, how to check a change |
| `docs/experiments/` | One dated record per decision, with its data |

A new model goes in `src/tt_models/models/<name>/`, with its adapter class
added to `MODELS` in `src/tt_models/vllm_plugin.py`. Code that two models share moves to
`common/`.

## Working on it

    pixi run pytest                # host tests
    pixi run pytest -m device      # device tests with random weights
    pixi run python scripts/accuracy/greedy_check.py --compare scripts/accuracy/greedy-base.json

[docs/serving.md](docs/serving.md) lists which check covers what. Results
go to `artifacts/` (not tracked) until they back a decision; then they move
into a new doc under [docs/experiments/](docs/experiments/README.md), and
README and guides link to it rather than quoting numbers.
`AGENTS.md` holds device behaviour seen on this machine but not yet
understood.
