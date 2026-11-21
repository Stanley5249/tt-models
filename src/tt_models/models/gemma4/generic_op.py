"""Helpers for this model's custom kernels (kernels/), run through ttnn.generic_op."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Iterable

KERNELS = Path(__file__).parent / "kernels"
BFP4_TILE = 576
BFP8_TILE = 1088
BF16_TILE = 2048
FP32_TILE = 4096


type ProgramParts = tuple[ttnn.ProgramDescriptor, list[ttnn.Tensor]]


def tile_bytes(dtype: ttnn.DataType) -> int:
    """Return the byte size of a tile with the given dtype."""
    return {
        ttnn.bfloat4_b: BFP4_TILE,
        ttnn.bfloat8_b: BFP8_TILE,
        ttnn.bfloat16: BF16_TILE,
        ttnn.float32: FP32_TILE,
    }[dtype]


def cb(
    cores: ttnn.CoreRangeSet, index: int, dtype: ttnn.DataType, page: int, pages: int
) -> ttnn.CBDescriptor:
    """A circular buffer of `pages` pages of `page` bytes on `cores`."""
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page
            )
        ],
    )


def accessor(t: ttnn.Tensor) -> list[int]:
    """Compile-time args of a TensorAccessor for t."""
    return ttnn.TensorAccessorArgs(t).get_compile_time_args()


def line_cores(n: int, y0: int = 0) -> list[ttnn.CoreCoord]:
    """N logical cores, 8 per row from row y0."""
    return [ttnn.CoreCoord(i % 8, y0 + i // 8) for i in range(n)]


def core_range_set(cores: Iterable[ttnn.CoreCoord]) -> ttnn.CoreRangeSet:
    """Build a core range set from individual coordinates."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def f32_bits(v: float) -> int:
    """A float as the uint32 bits a kernel's compile-time arg carries."""
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]
