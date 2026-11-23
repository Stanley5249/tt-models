"""Decode router op (kernels/router_*)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import torch


class Router:
    """Fused decode router for 1-16 tokens.

    Logits, top-k, softmax over the k logits and the step's distinct experts in one
    op (kernels/router_*).

    topk_router_gpt took 68-72 us per layer, mostly in its top-k stage. Here 8-11 cores
    each multiply their K slice of the normed rows by the BF16 router weight, the
    reducer core gathers the partial rows (semaphore), sums each token's rows and takes
    exp on the FPU/SFPU, and the cores' RISCs pick the tokens' top k and normalize them,
    a share of the rows each (scripts/kernels/router_check.py).
    Its weights are closer to an fp32 reference (max error 0.0007 against
    topk_router_gpt's 0.11).
    Outputs (row-major): uint16 ids of the step's distinct experts, one slot each, with
    their count at entry `slots`; BF16 weights slot-major [slot * rows + t], zero where
    token t did not pick the slot's expert. The expert streams read each slot's
    weights once for all its tokens.
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

        rows: decode tokens (1 .. 16). The reducer stacks every core's partial rows in
        one tile, so parts * rows must fit 32 and parts must divide the 88 K tiles:
        3-4 tokens use 8 parts of 11 K tiles (the first 8 of the 11 cores). With fewer
        parts than cores, each part also splits the expert tiles over up to 4 cores
        (16 tokens: 2 parts x 4), so no core reads more than a quarter of a K half.
        """
        self.mesh, self.top_k, self.rows = mesh, top_k, rows
        self.kx = hidden // TILE
        self.n_tiles = n_experts // TILE
        while parts * rows > TILE or self.kx % parts:
            parts -= 1
        self.parts = parts
        self.kt = self.kx // parts
        self.n_split = next(
            s for s in (4, 2, 1) if parts * s <= 11 and self.n_tiles % s == 0
        )
        self.tn = self.n_tiles // self.n_split  # expert tiles per core
        assert rows <= TILE // 2, "token rows must fit the top faces of a tile"
        self.slots = min(n_experts, rows * top_k)
        # Row lengths in entries, whole 64-byte lines: ids then the count, weights.
        self.id_entries = -(-(self.slots + 1) // TILE) * TILE
        self.wt_entries = -(-(self.slots * rows) // TILE) * TILE
        self.cores = [
            ttnn.CoreCoord(i % 8, core_row - i // 8)
            for i in range(parts * self.n_split)
        ]
        self.red = mesh.worker_core_from_logical_core(self.cores[0])

    def outputs(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Allocate the row-major expert ids and weights the expert streams take."""

        def row(n: int, dtype: ttnn.DataType) -> ttnn.Tensor:
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, n]), dtype, ttnn.ROW_MAJOR_LAYOUT, self.mesh, L1
            )

        return row(self.id_entries, ttnn.uint16), row(self.wt_entries, ttnn.bfloat16)

    def decode(self, ids: torch.Tensor, wts: torch.Tensor) -> list[dict[int, float]]:
        """Each token's {expert: weight} from host copies of the outputs (checks)."""
        ids, wts = ids.flatten().long(), wts.flatten().float()
        n = int(ids[self.slots])
        return [
            {
                int(ids[s]): float(wts[s * self.rows + t])
                for s in range(n)
                if wts[s * self.rows + t] != 0
            }
            for t in range(self.rows)
        ]

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
        cb_x, cb_w, cb_j, cb_partial, cb_part, cb_e, cb_out, cb_res = range(8)
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        n = self.n_tiles
        cbs = [
            cb(crs, cb_x, bf16, 2048, self.kt),
            cb(crs, cb_w, bf16, 2048, self.kt * self.tn),
            cb(crs, cb_j, f32, FP32_TILE, 1),
            cb(crs, cb_partial, f32, FP32_TILE, self.tn),
            cb(crs, cb_part, f32, FP32_TILE, n),
            cb(crs, cb_e, f32, FP32_TILE, n),
            cb(crs, cb_out, bf16, 2 * (self.id_entries + self.wt_entries), 1),
            cb(crs, cb_res, bf16, 4 * self.top_k * self.rows, 1),  # k ids, k weights
        ]
        # The reducer signals the top-k cores (router_writer.cpp) at these NOC coords.
        peers = [0] * (2 * 10)
        for r, c in enumerate(self.cores[1:]):
            noc = self.mesh.worker_core_from_logical_core(c)
            peers[2 * r : 2 * r + 2] = [noc.x, noc.y]
        reader_rt, writer_rt, compute_rt = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        for i, c in enumerate(self.cores):
            red = int(i == 0)
            part, n0 = i // self.n_split, (i % self.n_split) * self.tn
            reader_rt[c.x][c.y] = [
                x.buffer_address(),
                w.buffer_address(),
                part * self.kt,
                red,
                n0,
            ]
            writer_rt[c.x][c.y] = [
                part,
                self.red.x,
                self.red.y,
                red,
                ids.buffer_address(),
                wts.buffer_address(),
                n0,
                i,
                *peers,
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
                    self.tn,
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
                    self.slots,
                    2 * self.id_entries,
                    2 * self.wt_entries,
                    self.tn,
                    len(self.cores),
                    cb_res,
                    1,
                    2,
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
                    self.tn,
                ],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
            for i in (sem_id, 1, 2)  # partials, top-k ready, top-k done
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return (
            (program, [x, w, ids, wts])
            if build
            else ttnn.generic_op([x, w, ids, wts], program)
        )
