"""Decode router op (kernels/router_*)."""

from __future__ import annotations

import ttnn

from tt_models.common.config import L1, TILE
from tt_models.models.gemma4.generic_op import (
    FP32_TILE,
    KERNELS,
    ProgramParts,
    accessor,
    cb,
    core_range_set,
)


class Router:
    """Fused decode router for one to four tokens.

    Decode router for 1-4 tokens: logits, top-k and softmax over the k logits in one
    op (kernels/router_*).

    topk_router_gpt took 68-72 us per layer, mostly in its top-k stage. Here 11 cores (8
    at 3-4 tokens)
    each multiply their K slice of the normed rows by the BF16 router weight, the
    reducer core gathers the
    partial rows (semaphore), sums each token's rows and takes exp on the FPU/SFPU, and
    its RISC picks every
    token's top k and normalizes them: 37-42 us at 1-2 tokens, 57-68 us at 3-4
    (scripts/kernels/router_check.py).
    Its weights are closer to an fp32 reference (max error 0.0007 against
    topk_router_gpt's 0.11).
    Outputs: row-major uint16 ids and BF16 weights, token-major (entry t * k + j).
    """

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        hidden: int,
        n_experts: int,
        top_k: int,
        parts: int = 11,
        core_row: int = 5,
        rows: int = 1,
    ) -> None:
        """Assign router partials to cores and output rows.

        rows: decode tokens (1 .. 4). The reducer stacks every core's partial rows in
        one tile, so
        parts * rows must fit 32: 3-4 tokens use 8 parts of 11 K tiles (the first 8 of
        the 11 cores).
        """
        self.mesh, self.top_k, self.rows = mesh, top_k, rows
        self.kx = hidden // TILE
        self.n_tiles = n_experts // TILE
        while parts * rows > TILE or self.kx % parts:
            parts -= 1
        self.parts = parts
        self.kt = self.kx // parts
        assert rows * top_k <= TILE
        self.cores = [ttnn.CoreCoord(i % 8, core_row - i // 8) for i in range(parts)]
        self.red = mesh.worker_core_from_logical_core(self.cores[0])

    def outputs(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Allocate row-major expert ids and weights.

        Row-major ids and weights, token-major (entry t * k + j), at least 16 entries
        (32 bytes): the
        layout the expert streams take.
        """

        def row(dtype: ttnn.DataType) -> ttnn.Tensor:
            n = max(16, self.rows * self.top_k)
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, n]), dtype, ttnn.ROW_MAJOR_LAYOUT, self.mesh, L1
            )

        return row(ttnn.uint16), row(ttnn.bfloat16)

    def __call__(
        self,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        ids: ttnn.Tensor,
        wts: ttnn.Tensor,
        *,
        build: bool = False,
    ) -> ProgramParts | ttnn.Tensor:
        """Compute the router logits, ids and weights for the decode rows.

        x: normed row form BF16 [1, 1, rows, H]; w: the [H, E] BF16 router weight
        (tile layout).
        build: return (program, tensors) to run merged with another op
        (expert_stream.run).
        """
        crs = core_range_set(self.cores)
        cb_x, cb_w, cb_j, cb_partial, cb_part, cb_e, cb_out = 0, 1, 2, 3, 4, 5, 6
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        n = self.n_tiles
        cbs = [
            cb(crs, cb_x, bf16, 2048, self.kt),
            cb(crs, cb_w, bf16, 2048, self.kt * n),
            cb(crs, cb_j, f32, FP32_TILE, 1),
            cb(crs, cb_partial, f32, FP32_TILE, n),
            cb(crs, cb_part, f32, FP32_TILE, n),
            cb(crs, cb_e, f32, FP32_TILE, n),
            cb(crs, cb_out, bf16, 128, 1),
        ]
        reader_rt, writer_rt, compute_rt = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        for i, c in enumerate(self.cores):
            red = int(i == 0)
            reader_rt[c.x][c.y] = [
                x.buffer_address(),
                w.buffer_address(),
                i * self.kt,
                red,
            ]
            writer_rt[c.x][c.y] = [
                i,
                self.red.x,
                self.red.y,
                red,
                ids.buffer_address(),
                wts.buffer_address(),
            ]
            compute_rt[c.x][c.y] = [red]
        sem_id = 0
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_reader.cpp"),
                core_ranges=crs,
                compile_time_args=[
                    cb_x,
                    cb_w,
                    cb_j,
                    self.kt,
                    n,
                    self.rows,
                    *accessor(x),
                    *accessor(w),
                ],
                runtime_args=reader_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_writer.cpp"),
                core_ranges=crs,
                compile_time_args=[
                    cb_partial,
                    cb_part,
                    cb_e,
                    cb_out,
                    n,
                    self.parts,
                    self.top_k,
                    sem_id,
                    self.rows,
                    max(32, 2 * self.rows * self.top_k),
                    *accessor(ids),
                    *accessor(wts),
                ],
                runtime_args=writer_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "router_compute.cpp"),
                core_ranges=crs,
                compile_time_args=[
                    cb_x,
                    cb_w,
                    cb_j,
                    cb_partial,
                    cb_part,
                    cb_e,
                    self.kt,
                    n,
                ],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=sem_id, core_ranges=crs, initial_value=0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return (
            (program, [x, w, ids, wts])
            if build
            else ttnn.generic_op([x, w, ids, wts], program)
        )
