"""The vLLM general plugin: entry point, model registration and its precedence."""

from importlib.metadata import entry_points

import pytest
from vllm.model_executor.models.registry import ModelRegistry
from vllm.tool_parsers import ToolParserManager

from tt_models.vllm_plugin import MODELS, register

ARCH = "TTGemma4ForConditionalGeneration"
OTHER = "some.module:OtherClass"


def registered(arch):
    return ModelRegistry.models[arch]


@pytest.fixture(autouse=True)
def restore_registry(monkeypatch):
    """Each test seeds the registry through monkeypatch, which restores it afterwards."""
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))


def seed(arch, target):
    """Register `target` for `arch` as another plugin would have, before ours runs."""
    ModelRegistry.register_model(arch, target)


def test_entry_point_is_installed():
    eps = entry_points(group="vllm.general_plugins")
    assert any(ep.value == "tt_models.vllm_plugin:register" for ep in eps)


def test_register_adds_model_and_tool_parser(monkeypatch):
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    seed(ARCH, OTHER)  # as if another plugin had run first
    register()
    module, cls = MODELS[ARCH].split(":")
    assert (registered(ARCH).module_name, registered(ARCH).class_name) == (module, cls)
    assert "gemma4_tt" in ToolParserManager.list_registered()


@pytest.mark.parametrize("name", [ARCH, ARCH.removeprefix("TT")])
def test_explicit_override_wins(monkeypatch, name):
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", f"{name}={OTHER}")
    seed(ARCH, OTHER)  # what vllm-tt-plugin registers for the override
    register()
    assert (registered(ARCH).module_name, registered(ARCH).class_name) == (
        "some.module",
        "OtherClass",
    )
