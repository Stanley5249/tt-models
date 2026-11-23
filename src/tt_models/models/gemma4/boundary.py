"""Batch-1 decode boundaries on a compact hidden vector.

Batch-1 decode layer boundaries on a compact hidden vector (kernels/compact*,
tail_*).

At batch 1 a decode hidden vector is row 0 of 88 [32, 32] tiles, and every
op on it pays for 32 rows: an all-gather of the 88 tiles took 34 us (55 us for
the feed-forward's two vectors), and the post-attention and post-FF norms,
residual adds and layer scalar were ~25 ops (~150 us per layer) of mostly
fixed cost.

Here the residual stream and the all-reduced partial sums are "compact": the
2816 values as FP32 chunks of 16, 64 chunks per [32, 32] FP32 tile, so a vector
is 3 dense tiles (compact.hpp). The o_proj and down streams write that form, the
all-gather moves 3 or 6 tiles, and one tail op per boundary does the sum over
chips, every norm, the residual adds and the layer scalar on a few tiles, then
scatters the next matmul inputs back to row form (BF16).

The residual stream is FP32 (it was BF16).
"""

from __future__ import annotations

import ttnn

from tt_models.common.config import L1, TILE
from tt_models.models.gemma4.compact import CHUNK, compact_tiles
from tt_models.models.gemma4.generic_op import (
    FP32_TILE,
    KERNELS,
    accessor,
    cb,
    core_range_set,
    f32_bits,
    line_cores,
)


class Boundary:
    """The compact tensors and the two tail ops of one mesh (shared by the layers)."""

    def __init__(
        self, mesh: ttnn.MeshDevice, hidden: int, eps: float, tail_cores: int = 6
    ) -> None:
        """Set up the compact operators for one mesh."""
        self.mesh = mesh
        self.hidden = hidden
        self.chunks = hidden // CHUNK  # 176
        self.V = compact_tiles(hidden)  # 3
        self.eps = eps
        # Each token row's tail runs on its own group of up to tail_cores cores; 11-16
        # rows get fewer per row so the groups fit the 64-core grid.
        self.tail_cores = tail_cores

    def tail_groups(self, rows: int) -> list[tuple[int, int]]:
        """The chunk range each core of one row's group writes."""
        n = min(self.tail_cores, 64 // rows)
        per = -(-self.chunks // n)
        return [
            (min(i * per, self.chunks), min((i + 1) * per, self.chunks))
            for i in range(n)
        ]

    # -- tensors ---------------------------------------------------------------

    def compact(self, vectors: int = 1) -> ttnn.Tensor:
        """Allocate a compact tensor for the given number of vectors."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, vectors * self.V * TILE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            self.mesh,
            L1,
        )

    def row(self, rows: int = 1) -> ttnn.Tensor:
        """Allocate a row-form tensor for the given number of rows."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            L1,
        )

    # -- ops -------------------------------------------------------------------

    def _tail(
        self,
        kind: str,
        g: ttnn.Tensor,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        x_out: ttnn.Tensor,
        outs: list[ttnn.Tensor],
        w_vectors: int,
        scalar: float = 1.0,
        x_page0: int = 0,
        rows: int = 1,
    ) -> ttnn.Tensor:
        V = self.V
        groups = self.tail_groups(rows)
        cores = line_cores(len(groups) * rows, y0=0)  # core i: row i // len(groups)
        crs = core_range_set(cores)
        cb_g, cb_x, cb_w, cb_j, cb_sq, cb_rs, cb_r, cb_t0, cb_xo, cb_row0 = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            12,
            16,
        )
        vecs = (
            2 if kind == "FF_TAIL" else 1
        )  # gathered vectors per row: [mlp | experts] or attention
        f32 = ttnn.float32
        cbs = [
            cb(
                crs, cb_g, f32, FP32_TILE, 2 * vecs * V
            ),  # this core's row only (tail_reader.cpp)
            cb(crs, cb_x, f32, FP32_TILE, V),
            cb(crs, cb_w, f32, FP32_TILE, w_vectors * V),
            cb(crs, cb_j, f32, FP32_TILE, 1),
            cb(crs, cb_sq, f32, FP32_TILE, V),
            cb(crs, cb_rs, f32, FP32_TILE, 1),
            cb(crs, cb_r, f32, FP32_TILE, 1),
            cb(crs, cb_xo, f32, FP32_TILE, V),
        ]
        cbs += [cb(crs, cb_t0 + i, f32, FP32_TILE, V) for i in range(4)]
        cbs += [cb(crs, cb_row0 + i, ttnn.bfloat16, 2048, V) for i in range(len(outs))]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            row, j = divmod(i, len(groups))
            c0, c1 = groups[j]
            reader_rt[c.x][c.y] = [
                g.buffer_address(),
                x.buffer_address(),
                w.buffer_address(),
                x_page0,
                row,
            ]
            writer_rt[c.x][c.y] = [
                x_out.buffer_address(),
                1 if j == 0 else 0,
                c0,
                c1,
                row,
            ] + [r.buffer_address() for r in outs]
        reader_ct = [
            cb_g,
            cb_x,
            cb_w,
            cb_j,
            w_vectors * V,
            V,
            rows,
            vecs,
            *accessor(g),
            *accessor(x),
            *accessor(w),
        ]
        writer_ct = [cb_xo, cb_row0, V, len(outs), *accessor(x_out), *accessor(outs[0])]
        compute_ct = [
            cb_g,
            cb_x,
            cb_w,
            cb_j,
            cb_sq,
            cb_rs,
            cb_r,
            cb_t0,
            cb_xo,
            cb_row0,
            V,
        ]
        compute_ct += [
            f32_bits(1.0 / self.hidden),
            f32_bits(self.eps),
            f32_bits(scalar),
            vecs * V,
            w_vectors * V,
        ]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_reader.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_writer.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=[(kind, "1")],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([g, x, w, *outs, x_out], program)

    def attn_tail(
        self,
        g: ttnn.Tensor,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        x_out: ttnn.Tensor,
        n: ttnn.Tensor,
        mlp_in: ttnn.Tensor,
        exp_in: ttnn.Tensor,
        x_page0: int = 0,
        rows: int = 1,
    ) -> ttnn.Tensor:
        """Sum the attention partials and prepare feedforward inputs.

        g: all-gathered compact attention partials [1, 2, 32, rows*V*32]; x: compact
        residual (rows
        vectors); w: compact [post_attention | pre_feedforward | pre_feedforward_2]
        weights. Per row: writes
        x_out = x + rms(sum g) * w_post_attn (compact) and n = rms(x_out), n * w_pre_ff,
        n * w_pre_ff2 (row
        form BF16 [1, 1, rows, H]).
        """
        return self._tail(
            "ATTN_TAIL",
            g,
            x,
            w,
            x_out,
            [n, mlp_in, exp_in],
            3,
            x_page0=x_page0,
            rows=rows,
        )

    def ff_tail(
        self,
        g: ttnn.Tensor,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        x_out: ttnn.Tensor,
        h: ttnn.Tensor,
        scalar: float,
        rows: int = 1,
    ) -> ttnn.Tensor:
        """Sum the feedforward partials and prepare the next layer's input.

        g: all-gathered compact [mlp rows | expert rows] partials [1, 2, 32,
        2*rows*V*32]; w: compact
        [post_ff1 | post_ff2 | post_ff | next input norm]. Per row: writes x_out = (x +
        rms(rms(m) w1 +
        rms(e) w2) w_ff) * scalar (compact) and h = rms(x_out) * w_in (row form BF16 [1,
        1, rows, H]).
        """
        return self._tail("FF_TAIL", g, x, w, x_out, [h], 4, scalar, rows=rows)
