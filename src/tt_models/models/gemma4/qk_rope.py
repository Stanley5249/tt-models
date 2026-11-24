"""Fused q/k RMSNorm, head weights and RoPE for the decode QKV rows."""

from __future__ import annotations

import torch
import ttnn

from tt_models.models.gemma4.compact import (
    CHUNK,
    CHUNKS_PER_TILE,
    compact_tensor,
    to_compact,
)
from tt_models.models.gemma4.generic_op import (
    FP32_TILE,
    KERNELS,
    accessor,
    cb,
    core_range_set,
    f32_bits,
)

ROPE_TILES_PER_ROW = 4  # cosine and signed sine for the repeated and last tiles


def rope_partner(width: int, head_dim: int) -> torch.Tensor:
    """Find the rotate_half partner of every value in a fused QKV row.

    Index of every fused-row value's rotate_half partner (the other half of its
    head).
    """
    d = torch.arange(width)
    half = head_dim // 2
    return torch.where(d % head_dim < half, d + half, d - half)


class QkRope:
    """Fused q/k RMSNorm, head weights and RoPE for the decode QKV rows.

    q/k RMSNorm, head weights and RoPE of every fused QKV row in one op
    (kernels/qk_rope_*).

    One core per compact tile (4 heads of 256 or 2 of 512) and token row; past 64
    cores a core takes several rows of its tile. V heads have weight 1, cos 1, sin 0.
    Replaces reshape, rms_norm, the weight multiply, the rotate_half matmul, three
    RoPE eltwise ops and the reshape back (~60 us per layer at batch 1, ~120 us at 16
    rows).
    """

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        width: int,
        head_dim: int,
        eps: float,
    ) -> None:
        """Set up the fused RMSNorm and RoPE operation."""
        self.mesh, self.width, self.head_dim, self.eps = mesh, width, head_dim, eps
        self.V = width // (CHUNK * CHUNKS_PER_TILE)
        assert width == self.V * CHUNK * CHUNKS_PER_TILE
        self.partner = rope_partner(width, head_dim)
        self.sign = torch.where(torch.arange(head_dim) < head_dim // 2, -1.0, 1.0)

    def weights(self, w_row: torch.Tensor) -> ttnn.Tensor:
        """[W] per-value head weights -> compact [w | w permuted to partners]."""
        return compact_tensor(self.mesh, torch.stack([w_row, w_row[self.partner]]))

    def host_tables(
        self, cos: torch.Tensor, sin: torch.Tensor, rotated_heads: int
    ) -> torch.Tensor:
        """Pack a repeated RoPE tile pair and the last tile pair, including V."""
        values = CHUNK * CHUNKS_PER_TILE
        heads = values // self.head_dim
        rotated_tail = rotated_heads - (self.V - 1) * heads
        assert 0 <= rotated_tail <= heads  # V heads fit entirely in the last tile
        pair = torch.stack([cos, sin * self.sign], dim=1).unsqueeze(2)
        repeated = pair.expand(-1, -1, heads, -1)
        tail = repeated.clone()
        tail[:, 0, rotated_tail:] = 1.0
        tail[:, 1, rotated_tail:] = 0.0
        packed = torch.stack([repeated, tail], dim=1).reshape(-1, values)
        return to_compact(packed)[None, None]

    def __call__(
        self,
        x: ttnn.Tensor,
        cs: ttnn.Tensor,
        w: ttnn.Tensor,
        out: ttnn.Tensor,
        cs_page0: int = 0,
    ) -> ttnn.Tensor:
        """Apply fused RMSNorm and RoPE to the QKV row.

        x: FP32 row-form QKV matmul output [1, 1, rows, W]; out: BF16 row form [1, 1,
        rows, W]; cs: four packed RoPE tiles per row from tile cs_page0.
        """
        rows = x.shape[2]
        groups = min(rows, 64 // self.V)  # cores per compact tile
        cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(groups * self.V)]
        crs = core_range_set(cores)
        cb_x, cb_xr, cb_cs, cb_w, cb_a, cb_b, cb_sq, cb_s1, cb_r, cb_t = range(10)
        cb_out = 16
        f32 = ttnn.float32
        cbs = [
            cb(crs, i, f32, FP32_TILE, n)
            for i, n in (
                (cb_x, 1),
                (cb_xr, 1),
                (cb_cs, 2),
                (cb_w, 2),
                (cb_a, 1),
                (cb_b, 1),
                (cb_sq, 1),
                (cb_t, 2),  # each its own CB: one push and pop per row keeps it aligned
                (cb_s1, 1),
                (cb_r, 1),
            )
        ]
        cbs.append(cb(crs, cb_out, ttnn.bfloat16, 2048, 1))
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            g, t = divmod(i, self.V)
            mine = list(range(g, rows, groups))
            # A fixed-length row list: the program cache keeps the first call's
            # runtime-argument count for programs that differ only in their rows.
            listed = [*mine, *[0] * (16 - len(mine))]
            reader_rt[c.x][c.y] = [
                x.buffer_address(),
                cs.buffer_address(),
                w.buffer_address(),
                t,
                cs_page0,
                len(mine),
                *listed,
            ]
            writer_rt[c.x][c.y] = [out.buffer_address(), t, len(mine), *listed]
            compute_rt[c.x][c.y] = [len(mine)]
        head_chunks = self.head_dim // CHUNK
        reader_ct = [
            cb_x,
            cb_xr,
            cb_cs,
            cb_w,
            cb_a,
            cb_b,
            head_chunks,
            self.V,
            *accessor(x),
            *accessor(cs),
            *accessor(w),
        ]
        compute_ct = [
            cb_x,
            cb_xr,
            cb_cs,
            cb_w,
            cb_a,
            cb_b,
            cb_sq,
            cb_s1,
            cb_r,
            cb_out,
            f32_bits(1.0 / self.head_dim),
            f32_bits(self.eps),
            cb_t,
        ]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_reader.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_writer.cpp"),
                core_ranges=crs,
                compile_time_args=[cb_out, *accessor(out)],
                runtime_args=writer_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        return ttnn.generic_op(
            [x, cs, w, out],
            ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
        )
