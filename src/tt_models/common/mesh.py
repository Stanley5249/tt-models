"""Open and close the N300 1x2 mesh with the fabric that the all-reduce needs."""

import contextlib

import ttnn

_PER_MESH: list[dict] = []


def per_mesh_cache() -> dict:
    """A module-level cache of mesh-bound objects, keyed by id(mesh).

    open_n300 clears every such cache when it closes its mesh, because a later
    mesh can reuse the closed one's id while the cached objects still point at
    the old device. (Under vLLM the plugin owns one mesh per process.)
    """
    cache: dict = {}
    _PER_MESH.append(cache)
    return cache


@contextlib.contextmanager
def open_n300(trace_region_size: int = 0):
    """Open a mesh using the configured topology."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2), trace_region_size=trace_region_size, num_command_queues=1
    )
    try:
        yield mesh
    finally:
        for cache in _PER_MESH:
            cache.clear()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
