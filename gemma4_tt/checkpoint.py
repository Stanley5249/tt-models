"""Lazy, mmap-backed access to the Gemma 4 text weights in a HF snapshot.

The full BF16 checkpoint (about 52 GB) does not fit in host RAM, so tensors
are read one at a time from the safetensors files and never all at once.
"""

import json
from functools import cached_property
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
TEXT_PREFIX = "model.language_model."


def resolve_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """Return the local snapshot directory, without downloading weights."""
    path = Path(model)
    if path.is_dir():
        return path
    # Only metadata patterns, so a snapshot with weights still downloading resolves too.
    return Path(snapshot_download(model, local_files_only=True, allow_patterns=["*.json", "*.jinja", "*.md"]))


class Checkpoint:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.root = resolve_snapshot(model)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles = {}

    @cached_property
    def config(self):
        """The text config (Gemma4TextConfig)."""
        return AutoConfig.from_pretrained(self.root).get_text_config()

    def _handle(self, filename: str):
        if filename not in self._handles:
            self._handles[filename] = safe_open(self.root / filename, framework="pt")
        return self._handles[filename]

    def get(self, name: str) -> torch.Tensor:
        """Read one text-model tensor by its name below `model.language_model.`."""
        full = TEXT_PREFIX + name
        return self._handle(self.weight_map[full]).get_tensor(full)

    def layer(self, idx: int) -> dict[str, torch.Tensor]:
        """All tensors of decoder layer `idx`, keyed relative to the layer."""
        prefix = f"{TEXT_PREFIX}layers.{idx}."
        return {
            key[len(prefix) :]: self._handle(file).get_tensor(key)
            for key, file in self.weight_map.items()
            if key.startswith(prefix)
        }
