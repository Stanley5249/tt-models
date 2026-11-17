"""Lazy access to the Gemma 4 text weights in a HF snapshot.

The full BF16 checkpoint (about 52 GB) does not fit in host RAM, so tensors
are read one at a time. The 50 GB shard cannot be mmapped either: with the
kernel's heuristic overcommit, safetensors' private mapping larger than RAM
fails with ENOMEM. So the header is parsed here and each tensor is read with
a plain `readinto` at its offset.
"""

import json
import struct
from functools import cached_property
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
TEXT_PREFIX = "model.language_model."
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


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
        self._headers: dict[str, tuple[int, dict]] = {}

    @cached_property
    def config(self):
        """The text config (Gemma4TextConfig)."""
        return AutoConfig.from_pretrained(self.root).get_text_config()

    def _header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._headers:
            with open(self.root / filename, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                self._headers[filename] = (8 + size, json.loads(f.read(size)))
        return self._headers[filename]

    def _read(self, filename: str, key: str) -> torch.Tensor:
        base, header = self._header(filename)
        meta = header[key]
        start, end = meta["data_offsets"]
        buf = bytearray(end - start)
        with open(self.root / filename, "rb") as f:
            f.seek(base + start)
            f.readinto(buf)
        return torch.frombuffer(buf, dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"])

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
