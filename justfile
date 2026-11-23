set default-list
set dotenv-load
set positional-arguments

# Overridable from .env; see .env.example.
export MESH_DEVICE := env("MESH_DEVICE", "N300")
host := env("HOST", "127.0.0.1")
port := env("PORT", "8000")
qwen3 := "Qwen/Qwen3-8B"

# Install the runtime environment.
install:
    pixi install

# Chunked prefill with one decode step after each chunk, and one decode trace
# per active row count in the trace region: docs/serving.md and
# docs/experiments/2026-11-21-decode-interleave.md. decode_only selects the
# model-side sampler. The served name must match the client's model ID.
[doc("Serve Gemma 4 26B-A4B as model \"gemma4\"; extra vllm serve flags pass through.")]
serve-gemma4 *args:
    pixi run vllm serve google/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
        --host "{{ host }}" --port "{{ port }}" \
        --served-model-name gemma4 \
        --block-size 64 \
        --max-num-seqs 16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 2048 \
        --enable-auto-tool-choice --tool-call-parser gemma4_tt \
        --reasoning-parser gemma4 \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}' \
        --additional-config '{"tt": {"trace_region_size": 200000000, "sample_on_device_mode": "decode_only", "decode_interleave_prefill_steps": 1}}' \
        "$@"

# tt_transformers reads the model ID from HF_MODEL.
[doc("Serve tt_transformers' Qwen3-8B; extra vllm serve flags pass through.")]
[env("TT_CACHE_PATH", "cache/Qwen3-8B")]
serve-qwen3-8b *args:
    HF_MODEL="{{ qwen3 }}" pixi run vllm serve "{{ qwen3 }}" \
        --host "{{ host }}" --port "{{ port }}" \
        --block-size 64 \
        --max-num-seqs 32 \
        --max-model-len 32768 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --reasoning-parser qwen3 \
        "$@"

# Format every source file.
[parallel]
fmt: _fmt-ruff _fmt-oxfmt _fmt-just

# Format, then apply the linter's automatic fixes.
fix: fmt
    ruff check --fix --quiet

# Lint Python, warnings as errors.
lint:
    ruff check --quiet

# Type-check Python against the runtime environment.
typecheck:
    pixi run pyrefly check

# Fast local gate: format checks, lint and type check.
[parallel]
check: (_fmt-ruff "--check") (_fmt-oxfmt "--check") (_fmt-just "--check") lint typecheck

# Host tests by default; `-m device` runs the N300 ones.
test *args:
    pixi run pytest "$@"

# Everything CI would run: the gate and the host tests.
ci: check test

_fmt-ruff flag="":
    ruff format --quiet {{ flag }}

_fmt-oxfmt flag="":
    bunx oxfmt@latest {{ flag }}

_fmt-just flag="":
    just --fmt {{ flag }}
