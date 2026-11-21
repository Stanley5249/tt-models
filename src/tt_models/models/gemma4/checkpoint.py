"""Lazy access to the Gemma 4 text weights in a HF snapshot.

The full BF16 checkpoint (about 52 GB) does not fit in host RAM, so tensors
are read one at a time. The 50 GB shard cannot be mmapped either: with the
kernel's heuristic overcommit, safetensors' private mapping larger than RAM
fails with ENOMEM. So the header is parsed here and each tensor is read with
a plain `readinto` at its offset.
"""

from __future__ import annotations

import json
import struct
from functools import cached_property
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
TEXT_PREFIX = "model.language_model."
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def resolve_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """Return the local snapshot directory, without downloading weights."""
    path = Path(model)
    if path.is_dir():
        return path
    # Only metadata patterns, so a snapshot with weights still downloading resolves too.
    return Path(
        snapshot_download(
            model, local_files_only=True, allow_patterns=["*.json", "*.jinja", "*.md"]
        )
    )


def attention_dims(cfg: Gemma4TextConfig) -> dict[str, tuple[int, int]]:
    """(head_dim, num_kv_heads) per layer type.

    transformers >= 5.15 keeps the global layers' values in per_layer_config
    and refuses the old global attributes, such as cfg.head_dim.
    """
    dims = {}
    assert cfg.layer_types is not None
    assert cfg.per_layer_config is not None
    for i, kind in enumerate(cfg.layer_types):
        layer = cfg.per_layer_config[i]
        dims.setdefault(kind, (layer.head_dim, layer.num_key_value_heads))
    return dims


class _TensorHeader(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]


class WeightSource(Protocol):
    """Provide the configuration and weights used by the device model."""

    @property
    def config(self) -> Gemma4TextConfig:
        """Return the text configuration."""
        ...

    def get(self, name: str) -> torch.Tensor:
        """Read a weight by its name relative to the text model."""
        ...


class Checkpoint:
    """Read config and tensors from a Gemma 4 checkpoint."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Resolve the snapshot and index its tensor shards."""
        self.root = resolve_snapshot(model)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[int, dict[str, _TensorHeader]]] = {}

    @cached_property
    def config(self) -> Gemma4TextConfig:
        """The text config (Gemma4TextConfig)."""
        config = AutoConfig.from_pretrained(self.root).get_text_config()
        assert isinstance(config, Gemma4TextConfig)
        return config

    def _header(self, filename: str) -> tuple[int, dict[str, _TensorHeader]]:
        if filename not in self._headers:
            with (self.root / filename).open("rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                self._headers[filename] = (8 + size, json.loads(f.read(size)))
        return self._headers[filename]

    def _read(self, filename: str, key: str) -> torch.Tensor:
        base, header = self._header(filename)
        meta = header[key]
        start, end = meta["data_offsets"]
        buf = bytearray(end - start)
        with (self.root / filename).open("rb") as f:
            f.seek(base + start)
            f.readinto(buf)
        return torch.frombuffer(buf, dtype=_DTYPES[meta["dtype"]]).reshape(
            meta["shape"]
        )

    def get(self, name: str) -> torch.Tensor:
        """Read one text-model tensor by its name below `model.language_model.`."""
        full = TEXT_PREFIX + name
        return self._read(self.weight_map[full], full)

    def layer(self, idx: int) -> dict[str, torch.Tensor]:
        """All tensors of decoder layer `idx`, keyed relative to the layer."""
        prefix = f"{TEXT_PREFIX}layers.{idx}."
        return {
            key[len(prefix) :]: self._read(file, key)
            for key, file in self.weight_map.items()
            if key.startswith(prefix)
        }
