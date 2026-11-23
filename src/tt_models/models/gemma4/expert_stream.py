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

Several decode tokens (rows 0 .. B - 1 of every tile, B <= 16) share one op
pair: the matmuls compute all rows anyway. The router (router.py) lists the
step's distinct experts, one slot each, with a weight per token row that is zero
for tokens that did not pick it, so an expert shared by several tokens is read
once and the down's slot sum leaves each token's experts in its row. The slot
count is known only on the device: every kernel reads it from the ids row, and
the gate|up cores of a bank split the slots into contiguous runs.

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


# One chip's stream cores: per-core assignment tuples and their range set.
type Placement = tuple[list[tuple[ttnn.CoreCoord, int, int, int]], ttnn.CoreRangeSet]


def bankline_cores(
    mesh: ttnn.MeshDevice,
    banks: Sequence[int],
    per_bank: int,
    exclude: Sequence[ttnn.CoreCoord] = (),
    chip: ttnn.MeshCoordinate | None = None,
) -> Placement:
    """Choose logical cores near each DRAM bank's optimal reader.

    per_bank logical cores per DRAM bank, nearest (Manhattan) to the bank's optimal
    NOC0 reader on `chip` (default chip 0; the chips are harvested differently).
    """
    opt = mesh.get_optimal_dram_bank_to_logical_worker_assignment(
        ttnn.NOC.NOC_0, chip or ttnn.MeshCoordinate(0, 0)
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


def run_on_chips(parts: Sequence[ProgramParts]) -> ttnn.Tensor:
    """Run one program per chip (part c on mesh column c) over the same tensors.

    The chips are harvested differently, so each needs its own cores next to its DRAM
    banks; a program placed from chip 0's map ran slower on chip 1.
    """
    chip = ttnn.MeshCoordinate
    programs = {
        ttnn.MeshCoordinateRange(chip(0, c), chip(0, c)): program
        for c, (program, _) in enumerate(parts)
    }
    return ttnn.generic_op(parts[0][1], ttnn.MeshProgramDescriptor(programs))


def row_bytes(entries: int) -> int:
    """Bytes the kernels read of a row-major ids or weights row: whole 64-byte lines."""
    return -(-2 * entries // 64) * 64


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
    slots: int,
    readers: int,
    pairs: int,
    n_scale: int,
    rows: int = 1,
    row_weights: bool = False,
    split: tuple[int, int] | None = None,
    fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
    build: bool = False,
) -> ProgramParts | ttnn.Tensor:
    """Build the fused expert gate and up operation.

    act[slot, pair] = gelu_tanh(x Wg) * (x Wu) * weights[slot] * scale[ids[slot]]
    (kernels/expert_gate_up_*).

    assign: (core, bank, r, vc, rank, pair_j, act_pair) per core. act tile = slot *
    pairs + act_pair. ids: at most `slots` expert ids and their count at entry
    `slots`; the `readers` cores of one bank (ranks 0 .. readers - 1) split the slots.
    rows: decode tokens (tile rows). row_weights: weights is slot-major [slot * rows +
    row]; otherwise weights[slot] scales every row.
    split: None for the expert layout (contiguous (gate, up) per k, expert-major), or
    (k_stride_tiles,
    up_delta) for the dense layout (one-tile packets at row offsets pair_j and pair_j +
    up_delta).
    """
    cb_x, cb_w, cb_sc, cb_rs, cb_ws, cb_act = 0, 1, 2, 3, 4, 16
    cb_cnt = 5
    block_k = 11  # k-steps per weight block (22 tiles)
    blocks = kx // block_k
    tb = tile_bytes(w.dtype)
    id_bytes = row_bytes(slots + 1)
    wt_bytes = row_bytes(slots * rows if row_weights else slots)
    cbs = [
        cb(cores, cb_x, ttnn.bfloat16, BF16_TILE, kx),
        cb(cores, cb_w, w.dtype, tb, 3 * 2 * block_k),
        cb(cores, cb_sc, ttnn.bfloat16, BF16_TILE, 2 * -(-slots // readers)),
        cb(cores, cb_rs, ttnn.bfloat16, id_bytes, 1),
        cb(cores, cb_ws, ttnn.bfloat16, id_bytes + wt_bytes + 2 * n_scale, 1),
        cb(cores, cb_cnt, ttnn.uint32, 16, 1),
        cb(cores, cb_act, ttnn.bfloat16, BF16_TILE, 2),
    ]
    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, bank, _, vc, rank, pair_j, act_pair in assign:
        reader_rt[core.x][core.y] = [
            w.buffer_address(),
            ids.buffer_address(),
            bank,
            vc,
            rank,
            pair_j,
        ]
        writer_rt[core.x][core.y] = [
            x.buffer_address(),
            act.buffer_address(),
            ids.buffer_address(),
            weights.buffer_address(),
            scale.buffer_address(),
            act_pair,
            rank,
        ]
    expert_bytes = kx * 2 * tb
    k_stride, up_delta = split or (0, 0)
    reader_ct = [
        cb_w,
        cb_rs,
        slots,
        blocks,
        2 * block_k,
        2,
        tb,
        expert_bytes,
        int(split is not None),
        k_stride * tb,
        up_delta,
        id_bytes,
        readers,
    ]
    reader_ct += accessor(ids)
    writer_ct = [
        cb_x,
        cb_sc,
        cb_act,
        cb_ws,
        kx,
        cb_cnt,
        pairs,
        n_scale,
        rows,
        int(row_weights),
        id_bytes,
        wt_bytes,
        slots,
        readers,
    ]
    for t in (x, act, ids, weights, scale):
        writer_ct += accessor(t)
    compute_ct = [cb_x, cb_w, cb_sc, cb_act, cb_cnt, blocks, block_k]
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
) -> ProgramParts:
    """Build the streamed expert down program (run it with run_on_chips).

    out[col] = sum over slots e and K tiles k of act[e, k] . W_{ids[e]}[k, col]
    (kernels/expert_down_*). ids: at most `slots` expert ids and their count at entry
    `slots`.

    w: interleaved [*, slots' K, banks * bank_cols * 32]; bank b holds columns b + banks
    * j. out: a
    [1, 1, rows, H] BF16 row-form tensor, or with compact_chunk0 a compact FP32 tensor
    (boundary.py) whose
    vector for row b starts at chunk compact_chunk0 + b * (chunks per vector).
    """
    cb_in0, cb_w, cb_rs, cb_zero, cb_cnt, cb_scr, cb_out = 0, 1, 3, 4, 5, 6, 16
    tb = tile_bytes(w.dtype)
    id_bytes = row_bytes(slots + 1)
    compact = compact_chunk0 is not None
    out_dtype = ttnn.float32 if compact else ttnn.bfloat16
    cbs = [
        cb(
            cores, cb_in0, ttnn.bfloat16, BF16_TILE, 2 * kt
        ),  # streamed one slot at a time
        cb(cores, cb_w, w.dtype, tb, 3 * kt * cols),
        cb(cores, cb_rs, ttnn.bfloat16, id_bytes, 1),
        cb(cores, cb_zero, ttnn.bfloat16, 64, 1),
        cb(cores, cb_cnt, ttnn.uint32, 16, 1),
        cb(cores, cb_scr, ttnn.bfloat16, id_bytes, 1),
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
            ids.buffer_address(),
        ]
    reader_ct = [
        cb_w,
        cb_rs,
        slots,
        kt,
        cols,
        bank_cols,
        tb,
        id_bytes,
        *accessor(ids),
    ]
    writer_ct = [
        cb_in0,
        cb_out,
        cb_cnt,
        cols,
        banks,
        n_tiles,
        int(compact),
        compact_chunk0 or 0,
        cb_zero,
        kt,
        rows,
        chunks + pad,
        cb_scr,
        id_bytes,
        slots,
    ]
    writer_ct += accessor(act) + accessor(out) + accessor(ids)
    compute_ct = [cb_in0, cb_w, cb_out, cb_cnt, kt, cols]
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
    return program, [w, act, ids, out]


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
        assert rows <= TILE // 2, "token rows must fit the top faces of a tile"
        self.gate_up_readers = gate_up_readers
        self.slots = min(n_experts, top_k * rows)  # the most distinct experts per step
        self.kx = hidden // TILE  # 88
        self.pairs = expert_local // TILE  # 11 real (gate, up) column pairs per chip
        self.kt = self.pairs  # down K tiles
        self.n_tiles = hidden // TILE  # 88 real down columns
        self.bank_cols = -(-self.n_tiles // self.banks)  # 8
        self.down_width = self.bank_cols * self.banks * TILE  # 3072
        assert self.bank_cols % down_readers == 0
        self.gu_slots = -(-self.slots // gate_up_readers)  # the most one core takes
        self.down_cols = self.bank_cols // down_readers
        assert self.down_cols <= 4, "fp32 dest holds 4 tiles"
        # (assignment, cores) per chip, each next to its own DRAM banks
        self.gu_chips: list[
            tuple[
                list[tuple[ttnn.CoreCoord, int, int, int, int, int, int]],
                ttnn.CoreRangeSet,
            ]
        ] = []
        self.down_chips: list[Placement] = []
        for i in range(mesh.get_num_devices()):
            chip = ttnn.MeshCoordinate(0, i)
            gu, gu_cores = bankline_cores(
                mesh, list(range(self.pairs)), gate_up_readers, chip=chip
            )
            self.gu_chips.append(
                ([(c, b, r, v, r, 0, b) for c, b, r, v in gu], gu_cores)
            )
            self.down_chips.append(
                bankline_cores(mesh, list(range(self.banks)), down_readers, chip=chip)
            )
        # chip 0's placement, for expert_q4's ops
        self.gu_assign, self.gu_cores = self.gu_chips[0]
        self.down_assign, self.down_cores = self.down_chips[0]
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

        X [1, 1, rows, H] BF16 tiles; ids / weights: the router's rows (router.py);
        scale: row-major BF16 [1, E] per_expert_scale; w_gate_up: the interleaved [1, E,
        H, 2 * pad] BFP4 weight.
        """
        return run_on_chips(
            [
                gate_up_op(
                    assign,
                    cores,
                    x,
                    ids,
                    weights,
                    scale,
                    w_gate_up,
                    act,
                    kx=self.kx,
                    slots=self.slots,
                    readers=self.gate_up_readers,
                    pairs=self.pairs,
                    n_scale=self.n_experts,
                    rows=self.rows,
                    row_weights=True,
                    build=True,
                )
                for assign, cores in self.gu_chips
            ]
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
        return run_on_chips(
            [
                down_op(
                    assign,
                    cores,
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
                for assign, cores in self.down_chips
            ]
        )


class DownStream:
    """A dense [K, H] BFP8 projection of decode rows into compact FP32 vectors.

    Through the expert down kernels: the weight is stored with H padded to whole bank
    columns (pad_down, so bank b holds columns b + 12 j) and read as `slots` blocks of
    K tiles with ids 0 .. slots - 1, on 2 cores per bank placed per chip. The dense
    MLP down and attention's o_proj use it; writing compact vectors saves o_proj's
    separate pack op.
    """

    def __init__(
        self,
        mesh: ttnn.MeshDevice,
        k: int,
        hidden: int,
        slots: int,
        readers: int = 2,
        fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
    ) -> None:
        """Place the readers of a [k, hidden] weight read in `slots` K blocks."""
        self.banks = banks = mesh.dram_grid_size().x
        self.slots, self.fidelity = slots, fidelity
        self.n_tiles = hidden // TILE
        self.bank_cols = -(-self.n_tiles // banks)
        self.width = self.bank_cols * banks * TILE  # the padded hidden, 3072
        assert (k // TILE) % slots == 0, "slots must split K"
        assert self.bank_cols % readers == 0
        self.kt = k // TILE // slots
        self.cols = self.bank_cols // readers
        self.chips = [
            bankline_cores(
                mesh, list(range(banks)), readers, chip=ttnn.MeshCoordinate(0, i)
            )
            for i in range(mesh.get_num_devices())
        ]
        # an ids row as the router's: the slots, then their count
        self.ids = _const_row(
            mesh, [*range(slots), slots] + [0] * (31 - slots), ttnn.uint16
        )

    def __call__(
        self,
        act: ttnn.Tensor,
        w: ttnn.Tensor,
        out: ttnn.Tensor,
        compact_chunk0: int | None = 0,
        rows: int = 1,
    ) -> ttnn.Tensor:
        """Out = act @ w for the token rows of act [1, 1, rows, K].

        out: compact FP32 vectors (one per row) from chunk compact_chunk0, or with
        compact_chunk0=None a [1, 1, rows, H] BF16 row-form tensor.
        """
        return run_on_chips(
            [
                down_op(
                    assign,
                    cores,
                    act,
                    self.ids,
                    w,
                    out,
                    slots=self.slots,
                    kt=self.kt,
                    cols=self.cols,
                    banks=self.banks,
                    bank_cols=self.bank_cols,
                    n_tiles=self.n_tiles,
                    rows=rows,
                    compact_chunk0=compact_chunk0,
                    fidelity=self.fidelity,
                )
                for assign, cores in self.chips
            ]
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
        gu, self.gu_cores = bankline_cores(
            mesh, list(range(banks)), self.per_bank, exclude=exclude
        )
        # pair j of bank b: gate column b + 12 j -> h tile b + 12 j
        self.gu_assign = [(c, b, j, v, 0, j, b + banks * j) for c, b, j, v in gu]
        # 4 slots of 9 K tiles
        self.down_stream = DownStream(
            mesh, mlp_pad, hidden, down_slots, down_readers, fidelity
        )
        self.down_width = self.down_stream.width
        # an ids row as the router's: the slot, then the count
        self.ids0 = _const_row(mesh, [0, 1] + [0] * 30, ttnn.uint16)
        self.ones = _const_row(mesh, [1.0] * 32, ttnn.bfloat16)

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
            slots=1,
            readers=1,
            pairs=0,
            n_scale=16,
            rows=rows,
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
        vectors (one per row) from chunk compact_chunk0, or with compact_chunk0=None a
        [1, 1, rows, H] BF16 row-form tensor.
        """
        return self.down_stream(h, w_down, out, compact_chunk0, rows)


def pad_down(down: torch.Tensor, width: int) -> torch.Tensor:
    """Pad down-projection weights for the bank-aligned layout.

    [..., I, H] -> [..., I, width] with zero columns, for the bank-aligned down
    layout.
    """
    return torch.nn.functional.pad(down, (0, width - down.shape[-1]))
