set default-list
set dotenv-load
set positional-arguments

# Fail on a stale pixi.lock instead of re-solving it.
export PIXI_LOCKED := "true"

# Overridable from .env; see .env.example.
export MESH_DEVICE := env("MESH_DEVICE", "N300")
host := env("HOST", "127.0.0.1")
port := env("PORT", "8000")

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
        --max-num-seqs 4 \
        --max-model-len 32768 \
        --max-num-batched-tokens 2048 \
        --enable-auto-tool-choice --tool-call-parser gemma4_tt \
        --reasoning-parser gemma4 \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}' \
        --additional-config '{"tt": {"trace_region_size": 200000000, "sample_on_device_mode": "decode_only", "decode_interleave_prefill_steps": 1}}' \
        "$@"

# tt_transformers reads HF_MODEL, so keep it in sync with the model ID.
[doc("Serve tt_transformers' Qwen3-8B; extra vllm serve flags pass through.")]
[env("HF_MODEL", "Qwen/Qwen3-8B")]
[env("TT_CACHE_PATH", "cache/Qwen3-8B")]
serve-qwen3-8b *args:
    pixi run vllm serve Qwen/Qwen3-8B \
        --host "{{ host }}" --port "{{ port }}" \
        --block-size 64 \
        --max-num-seqs 32 \
        --max-model-len 32768 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --reasoning-parser qwen3 \
        "$@"

# Format every source file.
[parallel]
fmt: _fmt-ruff _fmt-tombi _fmt-oxfmt _fmt-just

# Format, then apply the linter's automatic fixes.
fix: fmt
    ruff check --fix --quiet

# Lint Python, warnings as errors.
lint *args:
    ruff check --quiet "$@"

# Type-check Python against the runtime environment.
typecheck *args:
    pixi run pyrefly check "$@"

# Fast local gate: format checks, lint and type check.
[parallel]
check: (_fmt-ruff "--check") (_fmt-tombi "--check") (_fmt-oxfmt "--check") (_fmt-just "--check") lint typecheck

# Host tests; pytest flags pass through.
test *args:
    pixi run pytest "$@"

# Device tests on the N300.
test-device *args:
    pixi run pytest -m device "$@"

# Everything CI would run: the gate, the lockfile and the host tests.
ci: check lock-check test

# Fail when pixi.lock does not match pyproject.toml.
lock-check:
    pixi lock --check --dry-run

_fmt-ruff *args:
    ruff format --quiet "$@"

_fmt-tombi *args:
    tombi format --quiet "$@" .

_fmt-oxfmt *args:
    bunx oxfmt "$@"

_fmt-just *args:
    just --fmt "$@"
