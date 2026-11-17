"""Open and close the N300 1x2 mesh with the fabric that the all-reduce needs."""

import contextlib

import ttnn


@contextlib.contextmanager
def open_n300(trace_region_size: int = 0):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=trace_region_size, num_command_queues=1)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
