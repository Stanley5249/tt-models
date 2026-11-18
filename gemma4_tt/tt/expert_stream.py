"""Batch-1 decode experts as two DRAM-streaming generic ops (kernels/expert_*.cpp).

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
into a [32, 8 * 352] tile tensor (row 0). down: out[col] = sum over slots and k
of act . Wd, the slot sum fused into the K accumulation.
"""

from pathlib import Path

import torch
import ttnn

KERNELS = Path(__file__).parent / "kernels"
TILE = 32
BFP4_TILE = 576
BF16_TILE = 2048


def _bank_cores(mesh, banks, per_bank):
    """per_bank logical cores per DRAM bank, nearest (Manhattan) to the bank's optimal NOC0 reader."""
    opt = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
    grid = mesh.compute_with_storage_grid_size()
    free = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    out = {b: [] for b in banks}
    for r in range(per_bank):  # one core per bank per round, so no bank takes all the near cores
        for b in banks:
            w = opt[b]
            x, y = min(free, key=lambda c: (abs(c[0] - w.x) + abs(c[1] - w.y), abs(c[1] - w.y), c))
            free.discard((x, y))
            out[b].append(ttnn.CoreCoord(x, y))
    placed = [(c, b, r) for b in banks for r, c in enumerate(out[b])]
    vcs = []
    for i, (core, bank, r) in enumerate(placed):
        vc = (bank + r) & 3
        for j in range(i):  # as the DeepSeek / gpt-oss streamers: distinct VCs along a row
            if placed[j][0].y == core.y and vcs[j] == vc:
                vc = (vc + 1) & 3
        vcs.append(vc)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in placed])
    return [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)], cores


def _cb(cores, index, dtype, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _kernels(cores, name, reader, writer, compute):
    """reader / writer / compute: (compile args, runtime args); compute runtime args unused."""
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
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
        ),
    ]


def _accessor(t):
    return ttnn.TensorAccessorArgs(t).get_compile_time_args()


class ExpertStream:
    """Routed experts of one layer for one decode token (batch 1)."""

    def __init__(self, mesh, hidden: int, expert_local: int, top_k: int, n_experts: int, gate_up_readers=4, down_readers=2):
        self.mesh = mesh
        self.banks = mesh.dram_grid_size().x
        self.hidden, self.top_k, self.n_experts = hidden, top_k, n_experts
        self.kx = hidden // TILE  # 88
        self.pairs = expert_local // TILE  # 11 real (gate, up) column pairs per chip
        self.kt = self.pairs  # down K tiles
        self.n_tiles = hidden // TILE  # 88 real down columns
        self.bank_cols = -(-self.n_tiles // self.banks)  # 8
        self.down_width = self.bank_cols * self.banks * TILE  # 3072
        assert top_k % gate_up_readers == 0 and self.bank_cols % down_readers == 0
        self.gu_slots = top_k // gate_up_readers
        self.down_cols = self.bank_cols // down_readers
        assert self.down_cols <= 4, "fp32 dest holds 4 tiles"
        self.gu_assign, self.gu_cores = _bank_cores(mesh, list(range(self.pairs)), gate_up_readers)
        self.down_assign, self.down_cores = _bank_cores(mesh, list(range(self.banks)), down_readers)
        self.act_mem = ttnn.L1_MEMORY_CONFIG
        self._gu_program = None
        self._down_program = None

    def act_tensor(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.top_k * self.pairs * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, self.act_mem
        )

    def out_tensor(self, memory_config):
        """[1, 1, 1, H]: the kernels write row 0 of every tile; padding rows stay unwritten."""
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, memory_config)

    def gate_up(self, x, ids, weights, scale, w_gate_up, act):
        """x [1, 1, 1, H] BF16 tiles; ids / weights: topk_router_gpt's row-major [1, k]; scale: row-major
        BF16 [1, E] per_expert_scale; w_gate_up: the interleaved [1, E, H, 2 * pad] BFP4 weight."""
        cb_x, cb_w, cb_sc, cb_rs, cb_ws, cb_act = 0, 1, 2, 3, 4, 16
        block_k = 11  # k-steps per weight block: 22 tiles, two 6336-byte packets
        blocks = self.kx // block_k
        cores = self.gu_cores
        cbs = [
            _cb(cores, cb_x, ttnn.bfloat16, BF16_TILE, self.kx),
            _cb(cores, cb_w, ttnn.bfloat4_b, BFP4_TILE, 3 * 2 * block_k),
            _cb(cores, cb_sc, ttnn.bfloat16, BF16_TILE, 2 * self.gu_slots),
            _cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
            _cb(cores, cb_ws, ttnn.bfloat16, 512, 1),
            _cb(cores, cb_act, ttnn.bfloat16, BF16_TILE, 2),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.gu_assign:
            slot0 = r * self.gu_slots
            reader_rt[core.x][core.y] = [w_gate_up.buffer_address(), ids.buffer_address(), bank, vc, slot0]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                act.buffer_address(),
                ids.buffer_address(),
                weights.buffer_address(),
                scale.buffer_address(),
                bank,
                slot0,
            ]
        expert_bytes = self.kx * 2 * BFP4_TILE
        reader_ct = [cb_w, cb_rs, self.gu_slots, blocks, 2 * block_k, 2, BFP4_TILE, expert_bytes] + _accessor(ids)
        writer_ct = [cb_x, cb_sc, cb_act, cb_ws, self.kx, self.gu_slots, self.pairs, self.n_experts]
        for t in (x, act, ids, weights, scale):
            writer_ct += _accessor(t)
        compute_ct = [cb_x, cb_w, cb_sc, cb_act, self.gu_slots, blocks, block_k]
        program = ttnn.ProgramDescriptor(
            kernels=_kernels(cores, "gate_up", (reader_ct, reader_rt), (writer_ct, writer_rt), compute_ct), semaphores=[], cbs=cbs
        )
        return ttnn.generic_op([w_gate_up, x, ids, weights, scale, act], program)

    def down(self, act, ids, w_down, out):
        """act: gate_up's output; w_down: the interleaved [1, E, 352, 3072] BFP4 weight; out [1, 1, 32, H]."""
        cb_in0, cb_w, cb_rs, cb_out = 0, 1, 3, 16
        cores = self.down_cores
        J = self.down_cols
        kin = self.top_k * self.kt
        cbs = [
            _cb(cores, cb_in0, ttnn.bfloat16, BF16_TILE, kin),
            _cb(cores, cb_w, ttnn.bfloat4_b, BFP4_TILE, 3 * self.kt * J),
            _cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
            _cb(cores, cb_out, ttnn.bfloat16, BF16_TILE, J),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.down_assign:
            j0 = r * J
            reader_rt[core.x][core.y] = [w_down.buffer_address(), ids.buffer_address(), bank, vc, j0]
            writer_rt[core.x][core.y] = [act.buffer_address(), out.buffer_address(), bank, j0]
        reader_ct = [cb_w, cb_rs, self.top_k, self.kt, J, self.bank_cols, BFP4_TILE] + _accessor(ids)
        writer_ct = [cb_in0, cb_out, kin, J, self.banks, self.n_tiles] + _accessor(act) + _accessor(out)
        compute_ct = [cb_in0, cb_w, cb_out, self.top_k, self.kt, J]
        program = ttnn.ProgramDescriptor(
            kernels=_kernels(cores, "down", (reader_ct, reader_rt), (writer_ct, writer_rt), compute_ct), semaphores=[], cbs=cbs
        )
        return ttnn.generic_op([w_down, act, ids, out], program)


def pad_down(down: torch.Tensor, width: int) -> torch.Tensor:
    """[E, I, H] -> [E, I, width] with zero columns, for the bank-aligned down layout."""
    return torch.nn.functional.pad(down, (0, width - down.shape[-1]))
