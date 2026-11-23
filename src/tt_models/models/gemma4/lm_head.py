"""The tied LM head, split over the vocabulary across chips.

Each chip streams its half of the tied embedding (BFP8, 131072 columns) with
generic-op kernels (kernels/lm_head_*), as the expert streams do: the columns
are padded to a multiple of the 12 DRAM banks, so bank b holds tile columns
b + 12 j of every K row, and 1-2 cores next to each bank (READERS) read its
columns in groups of 3 tiles, accumulate all 88 K tiles in FP32 dest and write
the FP32 logits straight into row-major rows. The ttnn version (16 DRAM-sharded matmuls,
an untilize each and a concat) placed its readers from chip 0's DRAM map, which
cost chip 1 0.5 ms per step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import ttnn

from tt_models.common.config import DRAM, TILE
from tt_models.common.norm import rms_norm
from tt_models.common.weights import WeightLoader, norm_weight
from tt_models.models.gemma4.expert_stream import bankline_cores, run_on_chips
from tt_models.models.gemma4.generic_op import (
    BF16_TILE,
    FP32_TILE,
    KERNELS,
    accessor,
    cb,
    tile_bytes,
)

if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    from tt_models.models.gemma4.checkpoint import WeightSource

# Cores per DRAM bank by token rows: two read faster, but past a few rows their 64-byte
# logit writes slow the reads (1 row: 1.68 against 1.79 ms; 16 rows: 1.92 against 1.78).
READERS = {rows: 2 if rows <= 4 else 1 for rows in range(1, 33)}
GROUP = 3  # output tiles per FP32 dest pass
BLOCK_K = 11  # K rows per streamed block


class LmHead:
    """Project hidden states through the tied embedding weights."""

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        loader: WeightLoader,
        ckpt: WeightSource,
        cfg: Gemma4TextConfig,
        dtype: ttnn.DataType,
    ) -> None:
        """Load the embedding weight in the bank layout and the final norm."""
        self.mesh = mesh
        self.cfg = cfg
        self.tp = mesh.get_num_devices()
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.final_norm = loader.put(
            "final_norm",
            lambda: norm_weight(ckpt.get("norm.weight")),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.banks = banks = mesh.dram_grid_size().x
        self.n_tiles = self.vocab // self.tp // TILE  # 4096 real column tiles per chip
        per_core = 2 * GROUP  # every READERS count divides the bank's columns
        self.bank_cols = -(-self.n_tiles // banks // per_core) * per_core  # 342
        width = banks * self.bank_cols * TILE
        per_chip = self.vocab // self.tp

        def weight() -> torch.Tensor:
            t = ckpt.get("embed_tokens.weight").T  # [H, V]
            # one padded copy: [chip 0's vocab | zeros | chip 1's vocab | zeros]
            out = t.new_zeros(1, 1, self.hidden, self.tp * width)
            for c in range(self.tp):
                out[0, 0, :, c * width : c * width + per_chip] = t[
                    :, c * per_chip : (c + 1) * per_chip
                ]
            return out

        self.weight = loader.put(f"lm_head_banks{width}", weight, dtype, shard_dim=-1)
        # Each chip's cores sit next to its own DRAM banks.
        self.placement = {
            (c, readers): bankline_cores(
                mesh, list(range(banks)), readers, chip=ttnn.MeshCoordinate(0, c)
            )
            for c in range(self.tp)
            for readers in set(READERS.values())
        }

    def _program(
        self, chip: int, x: ttnn.Tensor, out: ttnn.Tensor
    ) -> ttnn.ProgramDescriptor:
        readers = READERS[x.shape[2]]
        assign, cores = self.placement[chip, readers]
        cb_in0, cb_w, cb_out = 0, 1, 16
        kt = self.hidden // TILE
        tb = tile_bytes(self.weight.dtype)
        cbs = [
            cb(cores, cb_in0, ttnn.bfloat16, BF16_TILE, kt),
            cb(cores, cb_w, self.weight.dtype, tb, 3 * BLOCK_K * GROUP),
            cb(cores, cb_out, ttnn.float32, FP32_TILE, 2 * GROUP),
        ]
        groups = self.bank_cols // readers // GROUP
        reader_rt, writer_rt, compute_rt = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        for core, bank, r, vc in assign:
            j0 = r * groups * GROUP
            reader_rt[core.x][core.y] = [
                self.weight.buffer_address(),
                bank,
                vc,
                j0,
                groups,
            ]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                out.buffer_address(),
                bank,
                j0,
                groups,
            ]
            compute_rt[core.x][core.y] = [groups]
        reader_ct = [cb_w, kt, self.bank_cols, tb, BLOCK_K, GROUP]
        writer_ct = [cb_in0, cb_out, kt, x.shape[2], GROUP, self.banks, self.n_tiles]
        writer_ct += accessor(x) + accessor(out)
        compute_ct = [cb_in0, cb_w, cb_out, kt, BLOCK_K, GROUP]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lm_head_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lm_head_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "lm_head_compute.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def logits(self, x: ttnn.Tensor) -> torch.Tensor:
        """Compute and read the final logits."""
        rows = x.shape[2]
        if rows > TILE:
            # The stream op takes one tile of rows; longer inputs go a tile at a time.
            parts = []
            for start in range(0, rows, TILE):
                part = ttnn.slice(
                    x, [0, 0, start, 0], [1, 1, min(start + TILE, rows), self.hidden]
                )
                parts.append(
                    self.logits(part)
                )  # row slices are views of x; do not free them
            return torch.cat(parts)
        out = self.device(x)
        logits = self.read(out)
        out.deallocate(True)
        return logits

    def device(self, x: ttnn.Tensor, *, normed: bool = False) -> ttnn.Tensor:
        """Compute raw FP32 logits on the device.

        x: row form [1, 1, rows <= 32, H]. Returns raw FP32 logits, row-major [1, 1,
        rows, V / tp] per chip; the host applies the final softcap.
        """
        h = x if normed else rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        rows = x.shape[2]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, self.vocab // self.tp]),
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh,
            DRAM,
        )
        run_on_chips(
            [(self._program(c, h, out), [self.weight, h, out]) for c in range(self.tp)]
        )
        if not normed:
            h.deallocate(True)
        return out

    def read(self, out: ttnn.Tensor, *, softcap: bool = True) -> torch.Tensor:
        """Read and combine the per-chip logits.

        fp32 logits [rows, V] of a row-major [1, 1, rows, V/tp] per-chip output.
        (Copying into a kept host tensor and converting its shards was slower:
        1.0 ms against 0.66 ms for one row.)
        """
        logits = (
            ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
            .float()
            .reshape(-1, self.vocab)
        )
        return self.softcap(logits) if softcap else logits

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the final logit softcap in place and return the logits.

        Apply the final logit softcap in place (cap * tanh(x / cap)) and return
        logits.

        It costs 0.34 ms for one row of 262,144 logits and 1.35 ms for four, and it
        keeps the order, so greedy and top-k sampling can cap only their candidates.
        numpy ufuncs run on this thread: torch's pool sleeps during the device wait,
        and waking it once made the same softcap take 1 ms instead of 0.2 ms.
        """
        cap = self.cfg.final_logit_softcapping
        if cap:
            arr = logits.numpy()
            np.divide(arr, cap, out=arr)
            np.tanh(arr, out=arr)
            np.multiply(arr, cap, out=arr)
        return logits
