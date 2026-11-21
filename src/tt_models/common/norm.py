"""RMSNorm on decode and prefill rows."""

import ttnn

from tt_models.common.config import DRAM, HIFI4, TILE

_NORM_SHARD = {}


def rms_norm(x: ttnn.Tensor, weight, eps: float) -> ttnn.Tensor:
    """RMSNorm over the last dim of [1, 1, T, H].

    For decode (T <= 32) the interleaved op runs on few cores (63 us on
    [32, 2816]); width-sharding over 8 cores is 18.6 us with both conversions
    (scripts/bench_norm.py), so small inputs take that path.
    """
    rows, width = x.shape[2], x.shape[3]
    if rows > TILE or (width // TILE) % 8:
        return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=DRAM, compute_kernel_config=HIFI4)
    if width not in _NORM_SHARD:
        per = width // TILE // 8
        shard = ttnn.create_sharded_memory_config(
            shape=(TILE, per * TILE),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            subblock_w=next(w for w in (4, 2, 1) if per % w == 0),
            block_h=1,
            block_w=per,
            inplace=False,
        )
        _NORM_SHARD[width] = (shard, cfg)
    shard, cfg = _NORM_SHARD[width]
    xs = ttnn.to_memory_config(x, shard)
    y = ttnn.rms_norm(xs, weight=weight, epsilon=eps, program_config=cfg, memory_config=shard, compute_kernel_config=HIFI4)
    xs.deallocate(True)
    out = ttnn.to_memory_config(y, DRAM)
    y.deallocate(True)
    return out
