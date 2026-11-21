"""Open and close the N300 1x2 mesh with the fabric that the all-reduce needs."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_PER_MESH: list[Callable[[], None]] = []


def per_mesh_cache[K, V]() -> dict[K, V]:
    """A module-level cache of mesh-bound objects, keyed by id(mesh).

    open_n300 clears every such cache when it closes its mesh, because a later
    mesh can reuse the closed one's id while the cached objects still point at
    the old device. (Under vLLM the plugin owns one mesh per process.)
    """
    cache: dict[K, V] = {}
    _PER_MESH.append(cache.clear)
    return cache


@contextlib.contextmanager
def open_n300(trace_region_size: int = 0) -> Generator[ttnn.MeshDevice]:
    """Open a mesh using the configured topology."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2), trace_region_size=trace_region_size, num_command_queues=1
    )
    try:
        yield mesh
    finally:
        for clear in _PER_MESH:
            clear()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
