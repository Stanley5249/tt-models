"""Register models and parsers in each vLLM process.

vLLM general plugin: registers this package's models and parsers in every vLLM
process.

vLLM loads it through the `vllm.general_plugins` entry point in pyproject.toml, so
`vllm serve` needs no `--tool-parser-plugin` flag or `TT_MODEL_CLASS_OVERRIDES` entry.
vllm-tt-plugin's built-in classes register only where no class is registered yet, so
these win whichever plugin loads first. An explicit `TT_MODEL_CLASS_OVERRIDES` entry for
the same architecture still takes precedence.
"""

import os

MODELS = {
    (
        "TTGemma4ForConditionalGeneration"
    ): "tt_models.models.gemma4.vllm_adapter:Gemma4ForCausalLM",
}


def register() -> None:
    """Register the Gemma 4 models and tool parser with vLLM."""
    from vllm.model_executor.models.registry import ModelRegistry

    import tt_models.models.gemma4.tool_parser  # noqa: F401  registers the gemma4_tt parser

    overridden = {
        e.split("=", 1)[0].strip()
        for e in os.getenv("TT_MODEL_CLASS_OVERRIDES", "").split(",")
        if "=" in e
    }
    for arch, target in MODELS.items():
        if arch not in overridden and arch.removeprefix("TT") not in overridden:
            ModelRegistry.register_model(arch, target)
