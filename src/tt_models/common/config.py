"""Memory and compute configs shared by the models."""

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32


def compute_config(
    fidelity=ttnn.MathFidelity.HiFi2, *, fp32_acc=True, packer_l1_acc=True, approx=False
):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


HIFI4 = compute_config(ttnn.MathFidelity.HiFi4)
HIFI2 = compute_config(ttnn.MathFidelity.HiFi2)
LOFI = compute_config(ttnn.MathFidelity.LoFi)


def dram_sharded(mesh: ttnn.MeshDevice, k: int, n: int) -> ttnn.MemoryConfig:
    """Width-sharded DRAM config for a [k, n] decode weight, one shard per DRAM bank."""
    banks = mesh.dram_grid_size().x
    padded = -(-n // (TILE * banks)) * TILE * banks
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}
    )
    spec = ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec
    )


def width_sharded_l1(width: int, cores: int = 8) -> ttnn.MemoryConfig:
    """[32, width] width-sharded over cores (0, 0)..(cores - 1, 0)."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, width // cores),
        core_grid=ttnn.CoreGrid(x=cores, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
