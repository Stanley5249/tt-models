"""Decode experts as two DRAM-streaming generic ops (kernels/expert_*.cpp).

ttnn's indexed sparse_matmul splits only N over cores and reads the routed
experts tile by tile: 120 + 62 us per layer for the 8 routed experts' 13 MB
per chip (about 75 GB/s), plus ~15 glue ops for slicing, GELU, routing
weights and the slot sum. These ops instead stream each DRAM bank's share of
the routed experts in large packets, on cores next to that bank, after the
gpt-oss fused decode streamers (models/demos/gpt_oss/tt/fused_decode, which
are Blackhole-only; Wormhole has no custom_mm, so this uses matmul_tiles with
32x32 activation tiles whose row 0 is the token).

Layouts (per chip, interleaved BFP4, tile page p on DRAM bank p % 12):
* gate|up [128, 2816, 768]: column tiles 0..10 gate, 11 zero, 12..22 up, 23 zero.
  24 columns are a multiple of 12, so bank b holds the pair (gate b, up b) of
  every expert as one contiguous 176-tile block. This is the prefill weight
  itself; banks 0..10 hold real pairs.
* down [128, 352, 3072]: hidden padded to 96 column tiles, so bank b holds
  columns b + 12 j (j = 0..7), k-major, 88 tiles per expert.

gate_up: act[slot, pair] = gelu_tanh(x Wg) * (x Wu) * w_slot * per_expert_scale[e]
into a [32, slots * 352] tile tensor. down: out[col] = sum over slots and k of
act . Wd, the slot sum fused into the K accumulation.

Several decode tokens (rows 0 .. B - 1 of every tile) share one op pair: the
matmuls compute all rows anyway, and slot s = 8 * token + j scales only its
token's row, so the down's slot sum leaves each token's experts in its row.
The weight bytes grow with the B * 8 slots, the op count does not.

MlpStream runs the shared dense MLP through the same kernels: its per-chip
gate|up [2816, 2304] BFP8 has 72 column tiles, so bank b holds the 3 pairs
(gate b + 12 j, up 36 + b + 12 j); the down [1152, 3072] (hidden padded) is
read as 4 "slots" of 9 K tiles with ids 0..3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from tt_models.common.config import TILE
from tt_models.models.gemma4.generic_op import (
    BF16_TILE,
    KERNELS,
    ProgramParts,
    accessor,
    cb,
    tile_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _bankline_cores(
    mesh: ttnn.MeshDevice,
    banks: Sequence[int],
    per_bank: int,
    exclude: Sequence[ttnn.CoreCoord] = (),
) -> tuple[list[tuple[ttnn.CoreCoord, int, int, int]], ttnn.CoreRangeSet]:
    """Choose logical cores near each DRAM bank's optimal reader.

    per_bank logical cores per DRAM bank, nearest (Manhattan) to the bank's optimal
    NOC0 reader.
    """
    opt = mesh.get_optimal_dram_bank_to_logical_worker_assignment(
        ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0)
    )
    grid = mesh.compute_with_storage_grid_size()
    free = {(x, y) for x in range(grid.x) for y in range(grid.y)} - {
        (c.x, c.y) for c in exclude
    }
    out = {b: [] for b in banks}
    for _ in range(
        per_bank
    ):  # one core per bank per round, so no bank takes all the near cores
        for b in banks:
            w = opt[b]
            x, y = min(
                free,
                key=lambda c: (abs(c[0] - w.x) + abs(c[1] - w.y), abs(c[1] - w.y), c),
            )
            free.discard((x, y))
            out[b].append(ttnn.CoreCoord(x, y))
    placed = [(c, b, r) for b in banks for r, c in enumerate(out[b])]
    vcs = []
    for i, (core, bank, r) in enumerate(placed):
        vc = (bank + r) & 3
        for j in range(
            i
        ):  # as the DeepSeek / gpt-oss streamers: distinct VCs along a row
            if placed[j][0].y == core.y and vcs[j] == vc:
                vc = (vc + 1) & 3
        vcs.append(vc)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in placed])
    return [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)], cores


def _kernels(
    cores: ttnn.CoreRangeSet,
    name: str,
    reader: tuple[list[int], list[tuple[ttnn.CoreCoord, list[int]]]],
    writer: tuple[list[int], list[tuple[ttnn.CoreCoord, list[int]]]],
    compute: list[int],
    fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
) -> list[ttnn.KernelDescriptor]:
    """Build the reader, writer and compute kernel descriptors.

    Reader / writer / compute: (compile args, runtime args); compute runtime args
    unused.
    """
    dm = ttnn.DataMovementConfigDescriptor
    return [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / f"expert_{name}_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader[0],
            runtime_args=reader[1],
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / f"expert_{name}_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer[0],
            runtime_args=writer[1],
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / f"expert_{name}_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]


def _const_row(
    mesh: ttnn.MeshDevice, values: Sequence[int | float], dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Upload a row-major replicated constant.

    Row-major replicated [1, n] constant (the dense MLP's fixed ids, weights and
    scales).
    """
    return ttnn.from_torch(
        torch.tensor(
            [values], dtype=torch.float32 if dtype != ttnn.uint16 else torch.int32
        ),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def run(
    parts: Sequence[tuple[ttnn.ProgramDescriptor, list[ttnn.Tensor]]],
) -> ttnn.Tensor:
    """Run program and tensor pairs on disjoint cores as one generic op."""
    if len(parts) == 1:
        return ttnn.generic_op(parts[0][1], parts[0][0])
    tensors: list[ttnn.Tensor] = []
    seen: set[int] = set()
    for _, ios in parts:
        for t in ios:
            if id(t) not in seen:
                seen.add(id(t))
                tensors.append(t)
    return ttnn.generic_op(
        tensors, ttnn.merge_program_descriptors([p for p, _ in parts])
    )


def _id_bytes(slots: int) -> int:
    """Bytes the kernels read of the row-major ids / weights row (scratch holds 64)."""
    n = max(32, 2 * slots)
    assert n <= 64, "the kernels' scratch holds 32 slots"
    return n


def gate_up_op(
    assign: Sequence[tuple[ttnn.CoreCoord, int, int, int, int, int, int]],
    cores: ttnn.CoreRangeSet,
    x: ttnn.Tensor,
    ids: ttnn.Tensor,
    weights: ttnn.Tensor,
    scale: ttnn.Tensor,
    w: ttnn.Tensor,
    act: ttnn.Tensor,
    *,
    kx: int,
    slots_per_core: int,
    pairs: int,
    n_scale: int,
    rows: int = 1,
    slots_per_row: int = 0,
    split: tuple[int, int] | None = None,
    fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
    build: bool = False,
) -> ProgramParts | ttnn.Tensor:
    """Build the fused expert gate and up operation.

    act[slot, pair] = gelu_tanh(x Wg) * (x Wu) * weights[slot] * scale[ids[slot]]
    (kernels/expert_gate_up_*).

    assign: (core, bank, r, vc, slot0, pair_j, act_pair) per core. act tile = slot *
    pairs + act_pair.
    rows: decode tokens (tile rows); slot s scales row s // slots_per_row only, or every
    row with 0.
    split: None for the expert layout (contiguous (gate, up) per k, expert-major), or
    (k_stride_tiles,
    up_delta) for the dense layout (one-tile packets at row offsets pair_j and pair_j +
    up_delta).
    """
    cb_x, cb_w, cb_sc, cb_rs, cb_ws, cb_act = 0, 1, 2, 3, 4, 16
    block_k = 11  # k-steps per weight block (22 tiles)
    blocks = kx // block_k
    tb = tile_bytes(w.dtype)
    cbs = [
        cb(cores, cb_x, ttnn.bfloat16, BF16_TILE, kx),
        cb(cores, cb_w, w.dtype, tb, 3 * 2 * block_k),
        cb(cores, cb_sc, ttnn.bfloat16, BF16_TILE, 2 * slots_per_core),
        cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
        cb(cores, cb_ws, ttnn.bfloat16, 512, 1),
        cb(cores, cb_act, ttnn.bfloat16, BF16_TILE, 2),
    ]
    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, bank, _, vc, slot0, pair_j, act_pair in assign:
        reader_rt[core.x][core.y] = [
            w.buffer_address(),
            ids.buffer_address(),
            bank,
            vc,
            slot0,
            pair_j,
        ]
        writer_rt[core.x][core.y] = [
            x.buffer_address(),
            act.buffer_address(),
            ids.buffer_address(),
            weights.buffer_address(),
            scale.buffer_address(),
            act_pair,
            slot0,
        ]
    expert_bytes = kx * 2 * tb
    k_stride, up_delta = split or (0, 0)
    id_bytes = _id_bytes(max(a[4] for a in assign) + slots_per_core)
    reader_ct = [
        cb_w,
        cb_rs,
        slots_per_core,
        blocks,
        2 * block_k,
        2,
        tb,
        expert_bytes,
        int(split is not None),
        k_stride * tb,
        up_delta,
        id_bytes,
    ]
    reader_ct += accessor(ids)
    writer_ct = [
        cb_x,
        cb_sc,
        cb_act,
        cb_ws,
        kx,
        slots_per_core,
        pairs,
        n_scale,
        rows,
        slots_per_row,
        id_bytes,
    ]
    for t in (x, act, ids, weights, scale):
        writer_ct += accessor(t)
    compute_ct = [cb_x, cb_w, cb_sc, cb_act, slots_per_core, blocks, block_k]
    program = ttnn.ProgramDescriptor(
        kernels=_kernels(
            cores,
            "gate_up",
            (reader_ct, reader_rt),
            (writer_ct, writer_rt),
            compute_ct,
            fidelity,
        ),
        semaphores=[],
        cbs=cbs,
    )
    ios = [w, x, ids, weights, scale, act]
    return (program, ios) if build else ttnn.generic_op(ios, program)


def down_op(
    assign: Sequence[tuple[ttnn.CoreCoord, int, int, int]],
    cores: ttnn.CoreRangeSet,
    act: ttnn.Tensor,
    ids: ttnn.Tensor,
    w: ttnn.Tensor,
    out: ttnn.Tensor,
    *,
    slots: int,
    kt: int,
    cols: int,
    banks: int,
    bank_cols: int,
    n_tiles: int,
    rows: int = 1,
    compact_chunk0: int | None = None,
    fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
) -> ttnn.Tensor:
    """Build the streamed expert down operation.

    out[col] = sum over slots e and K tiles k of act[e, k] . W_{ids[e]}[k, col]
    (kernels/expert_down_*).

    w: interleaved [*, slots' K, banks * bank_cols * 32]; bank b holds columns b + banks
    * j. out: a
    [1, 1, rows, H] BF16 row-form tensor, or with compact_chunk0 a compact FP32 tensor
    (boundary.py) whose
    vector for row b starts at chunk compact_chunk0 + b * (chunks per vector).
    """
    cb_in0, cb_w, cb_rs, cb_zero, cb_out = 0, 1, 3, 4, 16
    tb = tile_bytes(w.dtype)
    compact = compact_chunk0 is not None
    out_dtype = ttnn.float32 if compact else ttnn.bfloat16
    cbs = [
        cb(
            cores, cb_in0, ttnn.bfloat16, BF16_TILE, 2 * kt
        ),  # streamed one slot at a time
        cb(cores, cb_w, w.dtype, tb, 3 * kt * cols),
        cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
        cb(cores, cb_zero, ttnn.bfloat16, 64, 1),
        cb(cores, cb_out, out_dtype, 2 * BF16_TILE if compact else BF16_TILE, cols),
    ]
    chunks = 2 * n_tiles
    pad = -(-chunks // 64) * 64 - chunks  # chunks + pad: one compact vector
    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (core, bank, r, vc) in enumerate(assign):
        j0 = r * cols
        reader_rt[core.x][core.y] = [
            w.buffer_address(),
            ids.buffer_address(),
            bank,
            vc,
            j0,
        ]
        zero = pad if compact and i == 0 else 0
        writer_rt[core.x][core.y] = [
            act.buffer_address(),
            out.buffer_address(),
            bank,
            j0,
            (compact_chunk0 or 0) + chunks,
            zero,
        ]
    reader_ct = [
        cb_w,
        cb_rs,
        slots,
        kt,
        cols,
        bank_cols,
        tb,
        _id_bytes(slots),
        *accessor(ids),
    ]
    writer_ct = [
        cb_in0,
        cb_out,
        slots,
        cols,
        banks,
        n_tiles,
        int(compact),
        compact_chunk0 or 0,
        cb_zero,
        kt,
        rows,
        chunks + pad,
    ]
    writer_ct += accessor(act) + accessor(out)
    compute_ct = [cb_in0, cb_w, cb_out, slots, kt, cols]
    program = ttnn.ProgramDescriptor(
        kernels=_kernels(
            cores,
            "down",
            (reader_ct, reader_rt),
            (writer_ct, writer_rt),
            compute_ct,
            fidelity,
        ),
        semaphores=[],
        cbs=cbs,
    )
    return ttnn.generic_op([w, act, ids, out], program)


class ExpertStream:
    """Stream one layer's routed experts for a batch of decode tokens.

    Routed experts of one layer for `rows` decode tokens (one op pair for all of
    them).
    """

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        hidden: int,
        expert_local: int,
        top_k: int,
        n_experts: int,
        gate_up_readers: int = 4,
        down_readers: int = 2,
        *,
        rows: int = 1,
    ) -> None:
        """Assign DRAM readers and tile work for routed experts."""
        self.mesh = mesh
        self.banks = mesh.dram_grid_size().x
        self.hidden, self.top_k, self.n_experts, self.rows = (
            hidden,
            top_k,
            n_experts,
            rows,
        )
        self.slots = top_k * rows
        self.kx = hidden // TILE  # 88
        self.pairs = expert_local // TILE  # 11 real (gate, up) column pairs per chip
        self.kt = self.pairs  # down K tiles
        self.n_tiles = hidden // TILE  # 88 real down columns
        self.bank_cols = -(-self.n_tiles // self.banks)  # 8
        self.down_width = self.bank_cols * self.banks * TILE  # 3072
        assert self.slots % gate_up_readers == 0
        assert self.bank_cols % down_readers == 0
        self.gu_slots = self.slots // gate_up_readers
        self.down_cols = self.bank_cols // down_readers
        assert self.down_cols <= 4, "fp32 dest holds 4 tiles"
        gu, self.gu_cores = _bankline_cores(
            mesh, list(range(self.pairs)), gate_up_readers
        )
        self.gu_assign = [(c, b, r, v, r * self.gu_slots, 0, b) for c, b, r, v in gu]
        self.down_assign, self.down_cores = _bankline_cores(
            mesh, list(range(self.banks)), down_readers
        )
        self.act_mem = ttnn.L1_MEMORY_CONFIG

    def act_tensor(self) -> ttnn.Tensor:
        """Allocate storage for the gate and up activations."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.slots * self.pairs * TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            self.act_mem,
        )

    def out_tensor(self, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        """Allocate row-form output for the expert stream.

        [1, 1, rows, H]: the kernels write the token rows of every tile; padding rows
        stay unwritten.
        """
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            memory_config,
        )

    def gate_up(
        self,
        x: ttnn.Tensor,
        ids: ttnn.Tensor,
        weights: ttnn.Tensor,
        scale: ttnn.Tensor,
        w_gate_up: ttnn.Tensor,
        act: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run the gate and up projections for the routed experts.

        X [1, 1, rows, H] BF16 tiles; ids / weights: row-major [1, rows * k],
        token-major (slot 8 t + j);
        scale: row-major BF16 [1, E] per_expert_scale; w_gate_up: the interleaved [1, E,
        H, 2 * pad] BFP4 weight.
        """
        return gate_up_op(
            self.gu_assign,
            self.gu_cores,
            x,
            ids,
            weights,
            scale,
            w_gate_up,
            act,
            kx=self.kx,
            slots_per_core=self.gu_slots,
            pairs=self.pairs,
            n_scale=self.n_experts,
            rows=self.rows,
            slots_per_row=self.top_k,
        )

    def down(
        self,
        act: ttnn.Tensor,
        ids: ttnn.Tensor,
        w_down: ttnn.Tensor,
        out: ttnn.Tensor,
        compact_chunk0: int | None = None,
    ) -> ttnn.Tensor:
        """act: gate_up's output; w_down: the interleaved [1, E, 352, 3072] BFP4 weight.

        out: a [1, 1, rows, H] BF16 row-form tensor (out_tensor), or with compact_chunk0
        a compact FP32
        tensor (boundary.py) whose vectors from chunk compact_chunk0 receive the rows'
        sums, padding zeroed.
        """
        return down_op(
            self.down_assign,
            self.down_cores,
            act,
            ids,
            w_down,
            out,
            slots=self.slots,
            kt=self.kt,
            cols=self.down_cols,
            banks=self.banks,
            bank_cols=self.bank_cols,
            n_tiles=self.n_tiles,
            rows=self.rows,
            compact_chunk0=compact_chunk0,
        )


class MlpStream:
    """Run one layer's shared dense MLP through the expert stream kernels.

    The shared dense MLP of one layer for up to 4 decode tokens, through the expert
    stream kernels.
    """

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        hidden: int,
        mlp_pad: int,
        down_readers: int = 2,
        down_slots: int = 4,
        fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        exclude: Sequence[ttnn.CoreCoord] = (),
    ) -> None:
        """Assign cores for the dense MLP stream.

        exclude: cores left free for an op that runs merged with gate_up (the
        router).
        """
        self.mesh = mesh
        self.fidelity = (
            fidelity  # BFP8 weights; the ttnn MLP used HiFi2 (LoFi: PCC 0.99985 vs it)
        )
        self.banks = banks = mesh.dram_grid_size().x
        self.hidden = hidden
        self.kx = hidden // TILE  # 88
        half = mlp_pad // TILE  # 36 gate (and 36 up) column tiles per chip
        assert half % banks == 0, (
            "gate|up column pairs must split evenly over the DRAM banks"
        )
        self.per_bank = half // banks  # 3 pairs per bank
        self.row_tiles = 2 * self.per_bank  # 6 tiles per bank per K row
        self.h_tiles = half
        gu, self.gu_cores = _bankline_cores(
            mesh, list(range(banks)), self.per_bank, exclude=exclude
        )
        # pair j of bank b: gate column b + 12 j -> h tile b + 12 j
        self.gu_assign = [(c, b, j, v, 0, j, b + banks * j) for c, b, j, v in gu]
        self.n_tiles = hidden // TILE
        self.bank_cols = -(-self.n_tiles // banks)
        self.down_width = self.bank_cols * banks * TILE
        assert half % down_slots == 0
        self.down_slots, self.down_kt = (
            down_slots,
            half // down_slots,
        )  # 4 slots of 9 K tiles
        self.down_cols = self.bank_cols // down_readers
        self.down_assign, self.down_cores = _bankline_cores(
            mesh, list(range(banks)), down_readers
        )
        self.ids0 = _const_row(mesh, [0] * 16, ttnn.uint16)
        self.ids_down = _const_row(
            mesh, list(range(down_slots)) + [0] * (16 - down_slots), ttnn.uint16
        )
        self.ones = _const_row(mesh, [1.0] * 16, ttnn.bfloat16)

    def h_tensor(self, rows: int = 1) -> ttnn.Tensor:
        """Allocate storage for the dense MLP activations."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, self.h_tiles * TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.L1_MEMORY_CONFIG,
        )

    def out_tensor(self, rows: int) -> ttnn.Tensor:
        """[1, 1, rows, H] row form, for down without compact_chunk0."""
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.L1_MEMORY_CONFIG,
        )

    def gate_up(
        self,
        x: ttnn.Tensor,
        w_gate_up: ttnn.Tensor,
        h: ttnn.Tensor,
        rows: int = 1,
        *,
        build: bool = False,
    ) -> ProgramParts | ttnn.Tensor:
        """Compute the dense MLP's gate and up activations.

        H = gelu_tanh(x Wg) * (x Wu) for the token rows of x; w_gate_up: the
        interleaved per-chip
        [2816, 2 * 1152] BFP8 weight.
        """
        return gate_up_op(
            self.gu_assign,
            self.gu_cores,
            x,
            self.ids0,
            self.ones,
            self.ones,
            w_gate_up,
            h,
            kx=self.kx,
            slots_per_core=1,
            pairs=0,
            n_scale=16,
            rows=rows,
            slots_per_row=0,
            split=(self.row_tiles, self.per_bank),
            fidelity=self.fidelity,
            build=build,
        )

    def down(
        self,
        h: ttnn.Tensor,
        w_down: ttnn.Tensor,
        out: ttnn.Tensor,
        compact_chunk0: int | None = 0,
        rows: int = 1,
    ) -> ttnn.Tensor:
        """Compute the dense MLP's down projection.

        Out = h @ W_down; w_down: [1152, 3072] BFP8 per chip. out: compact FP32
        vectors (one per row) from
        chunk compact_chunk0, or with compact_chunk0=None a [1, 1, rows, H] BF16
        row-form tensor.
        """
        return down_op(
            self.down_assign,
            self.down_cores,
            h,
            self.ids_down,
            w_down,
            out,
            slots=self.down_slots,
            kt=self.down_kt,
            cols=self.down_cols,
            banks=self.banks,
            bank_cols=self.bank_cols,
            n_tiles=self.n_tiles,
            rows=rows,
            compact_chunk0=compact_chunk0,
            fidelity=self.fidelity,
        )


def pad_down(down: torch.Tensor, width: int) -> torch.Tensor:
    """Pad down-projection weights for the bank-aligned layout.

    [..., I, H] -> [..., I, width] with zero columns, for the bank-aligned down
    layout.
    """
    return torch.nn.functional.pad(down, (0, width - down.shape[-1]))
