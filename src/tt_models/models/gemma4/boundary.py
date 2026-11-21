"""Batch-1 decode layer boundaries on a compact hidden vector (kernels/compact*, tail_*).

At batch 1 a decode hidden vector is row 0 of 88 [32, 32] tiles, and every
op on it pays for 32 rows: an all-gather of the 88 tiles took 34 us (55 us for
the feed-forward's two vectors), and the post-attention and post-FF norms,
residual adds and layer scalar were ~25 ops (~150 us per layer) of mostly
fixed cost.

Here the residual stream and the all-reduced partial sums are "compact": the
2816 values as FP32 chunks of 16, 64 chunks per [32, 32] FP32 tile, so a vector
is 3 dense tiles (compact.hpp). Matmul outputs are packed into that form, the
all-gather moves 3 or 6 tiles, and one tail op per boundary does the sum over
chips, every norm, the residual adds and the layer scalar on a few tiles, then
scatters the next matmul inputs back to row form (BF16).

The residual stream is FP32 (it was BF16).
"""

import ttnn

from tt_models.common.config import L1, TILE
from tt_models.models.gemma4.compact import CHUNK, CHUNKS_PER_TILE, compact_tiles
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
    """Compact pack and the two tail ops of one mesh (core placement shared by the layers)."""

    def __init__(self, mesh, hidden: int, eps: float, tail_cores: int = 6):
        self.mesh = mesh
        self.hidden = hidden
        self.chunks = hidden // CHUNK  # 176
        self.V = compact_tiles(hidden)  # 3
        self.eps = eps
        self.tail_cores = (
            tail_cores  # per token row: each row's tail runs on its own group of cores
        )
        per = -(-self.chunks // tail_cores)
        self.tail_ranges = [
            (min(i * per, self.chunks), min((i + 1) * per, self.chunks))
            for i in range(tail_cores)
        ]

    # -- tensors ---------------------------------------------------------------

    def compact(self, vectors: int = 1) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, vectors * self.V * TILE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            self.mesh,
            L1,
        )

    def row(self, rows: int = 1) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            L1,
        )

    # -- ops -------------------------------------------------------------------

    def pack(
        self,
        src: ttnn.Tensor,
        dst: ttnn.Tensor,
        cores: list,
        dst_chunk0: int = 0,
        rows: int = 1,
    ) -> ttnn.Tensor:
        """Row form FP32 [1, 1, rows, H] (any layout) -> compact vectors (one per row) from chunk dst_chunk0 of
        dst; zeroes the padding."""
        n_tiles = self.hidden // TILE
        per = -(-n_tiles // len(cores))
        crs = core_range_set(cores)
        rt = ttnn.RuntimeArgs()
        pad0 = dst_chunk0 + self.chunks
        pad = self.V * CHUNKS_PER_TILE - self.chunks
        for i, c in enumerate(cores):
            t0 = min(i * per, n_tiles)
            t1 = min(t0 + per, n_tiles)
            last = i == len(cores) - 1
            rt[c.x][c.y] = [
                src.buffer_address(),
                dst.buffer_address(),
                t0,
                t1 - t0,
                dst_chunk0,
                pad0,
                pad if last else 0,
            ]
        cb_scr = 0
        scr_bytes = (2 * per * rows + 1) * 64
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "compact_pack.cpp"),
            core_ranges=crs,
            compile_time_args=[cb_scr, rows, self.V * CHUNKS_PER_TILE]
            + accessor(src)
            + accessor(dst),
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
            ),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[kernel],
            semaphores=[],
            cbs=[cb(crs, cb_scr, ttnn.float32, scr_bytes, 1)],
        )
        return ttnn.generic_op([src, dst], program)

    def _tail(
        self,
        kind: str,
        g,
        x,
        w,
        x_out,
        outs: list,
        w_vectors: int,
        scalar: float = 1.0,
        x_page0: int = 0,
        rows: int = 1,
    ):
        V = self.V
        cores = line_cores(self.tail_cores * rows, y0=0)  # core i: row i // tail_cores
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
            row, j = divmod(i, self.tail_cores)
            c0, c1 = self.tail_ranges[j]
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
        reader_ct = (
            [cb_g, cb_x, cb_w, cb_j, w_vectors * V, V, rows, vecs]
            + accessor(g)
            + accessor(x)
            + accessor(w)
        )
        writer_ct = [cb_xo, cb_row0, V, len(outs)] + accessor(x_out) + accessor(outs[0])
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
        self, g, x, w, x_out, n, mlp_in, exp_in, x_page0: int = 0, rows: int = 1
    ):
        """g: all-gathered compact attention partials [1, 2, 32, rows*V*32]; x: compact residual (rows
        vectors); w: compact [post_attention | pre_feedforward | pre_feedforward_2] weights. Per row: writes
        x_out = x + rms(sum g) * w_post_attn (compact) and n = rms(x_out), n * w_pre_ff, n * w_pre_ff2 (row
        form BF16 [1, 1, rows, H])."""
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

    def ff_tail(self, g, x, w, x_out, h, scalar: float, rows: int = 1):
        """g: all-gathered compact [mlp rows | expert rows] partials [1, 2, 32, 2*rows*V*32]; w: compact
        [post_ff1 | post_ff2 | post_ff | next input norm]. Per row: writes x_out = (x + rms(rms(m) w1 +
        rms(e) w2) w_ff) * scalar (compact) and h = rms(x_out) * w_in (row form BF16 [1, 1, rows, H])."""
        return self._tail("FF_TAIL", g, x, w, x_out, [h], 4, scalar, rows=rows)
