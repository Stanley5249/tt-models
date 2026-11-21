"""Q4_0 decode experts: the checkpoint's 4-bit integers and learned steps, not BFP4
roundings.

The QAT checkpoint stores every 32-input block of an expert weight as q * s: integers q
in -8..7 and
one learned BF16 step s per (block, output). BFP4 (a power-of-two step per 16 outputs)
cannot hold
that grid (20 dB weight SNR). Here the device keeps q as BFP4 tiles, which hold -7..7
exactly (the
rare -8, 0.2% of the values, is clamped to -7: 35 dB), and s as BF16 step tiles. The
arithmetic is in
kernels/expert_q4_gate_up_compute.cpp: a 32-input block is one K tile, so the
activation's K tile k
sits in row k % 32 of a zero tile, each block's partial sum lands in its own dest row,
and an SFPU
multiply by the step tile plus a ones-row matmul finish the sum.

Experimental (Precision.expert_q4): batch-1 decode only. Decode accuracy against fp32
improves from
KL 0.457 to 0.286 (docs/experiments/2026-11-21-q4-decode-experts.md), at +27% gate|up
and +49% down device time. Prefill
has no Q4 path: ttnn's sparse matmuls cannot apply the steps, and two copies of the
experts do not
fit in DRAM, so moe.FeedForward runs the batch-1 streams token by token, which is only
good for
accuracy runs. The down step tiles keep 11 of 32 rows (25 MB per layer per chip);
serving would
store them compactly.
"""

import torch
import ttnn

from tt_models.common.config import TILE
from tt_models.models.gemma4.expert_stream import _id_bytes, pad_down
from tt_models.models.gemma4.generic_op import (
    BF16_TILE,
    KERNELS,
    accessor,
    cb,
    tile_bytes,
)
from tt_models.models.gemma4.q4 import BLOCK, factor_blocks

KERNELS_Q4 = "expert_q4_gate_up"


def q4_factor(
    w: torch.Tensor, dim: int = -2, chunk: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor w into integers q (clamped to -7..7) and steps s along 32-blocks of `dim`.

    Returns (q as float with w's shape, s with dim shrunk 32x); see q4.factor_blocks.
    Works on
    `chunk` slices of dim 0 at a time to bound host memory."""
    if w.dim() > 2 and w.shape[0] > chunk:
        parts = [
            q4_factor(w[i : i + chunk], dim, chunk) for i in range(0, w.shape[0], chunk)
        ]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    w = w.float().movedim(dim, -1)
    q, s = factor_blocks(w.reshape(*w.shape[:-1], -1, BLOCK))
    q = q.clamp(-7, 7)
    return q.reshape(w.shape).movedim(-1, dim), s.squeeze(-1).movedim(-1, dim)


def step_tiles(s: torch.Tensor, groups: int) -> torch.Tensor:
    """[..., E, K/32, N] steps -> [..., E, groups * 32, N]: row r of group g is block 32
    g + r, zero past K.
    """
    kb = s.shape[-2]
    return torch.nn.functional.pad(s, (0, 0, 0, groups * TILE - kb))


def gate_up_q4_op(
    assign,
    cores,
    x,
    ids,
    weights,
    scale,
    w,
    steps,
    act,
    *,
    kx,
    slots_per_core,
    pairs,
    n_scale,
    rows=1,
    slots_per_row=8,
):
    """gate|up from Q4_0 tiles (kernels/expert_q4_gate_up_*): as
    expert_stream.gate_up_op, with the step
    tensor steps ([1, E, groups * 32, N] BF16, interleaved). Each core's slots must
    belong to one token
    (slot s to token s // slots_per_row)."""
    cb_x, cb_w, cb_sc, cb_rs, cb_ws, cb_m, cb_r, cb_one, cb_act = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        16,
    )
    block_k = 11  # the stock reader's blocks (expert_gate_up_reader.cpp)
    assert kx % block_k == 0, f"{kx} K tiles do not split into blocks of {block_k}"
    blocks = kx // block_k
    groups = -(-kx // TILE)
    tb = tile_bytes(w.dtype)
    cbs = [
        cb(cores, cb_x, ttnn.bfloat16, BF16_TILE, kx),
        cb(cores, cb_w, w.dtype, tb, 3 * 2 * block_k),
        cb(cores, cb_sc, ttnn.bfloat16, BF16_TILE, 2 * slots_per_core),
        cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
        cb(cores, cb_ws, ttnn.bfloat16, 512, 1),
        cb(cores, cb_m, ttnn.bfloat16, BF16_TILE, 2 * groups * slots_per_core),
        cb(cores, cb_r, ttnn.float32, 2 * BF16_TILE, 2 * groups),
        cb(cores, cb_one, ttnn.bfloat16, BF16_TILE, 1),
        cb(cores, cb_act, ttnn.bfloat16, BF16_TILE, 2),
    ]
    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, bank, _, vc, slot0, pair_j, act_pair in assign:
        row = slot0 // slots_per_row
        assert (slot0 + slots_per_core - 1) // slots_per_row == row, (
            "a core's slots must belong to one token"
        )
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
            steps.buffer_address(),
            bank,
            row,
        ]
    id_bytes = _id_bytes(max(a[4] for a in assign) + slots_per_core)
    reader_ct = [
        cb_w,
        cb_rs,
        slots_per_core,
        blocks,
        2 * block_k,
        2,
        tb,
        kx * 2 * tb,
        0,
        0,
        0,
        id_bytes,
        *accessor(ids),
    ]
    writer_ct = [
        cb_x,
        cb_sc,
        cb_act,
        cb_ws,
        kx,
        slots_per_core,
        pairs,
        n_scale,
        id_bytes,
        cb_m,
        cb_one,
        rows,
    ]
    for t in (x, act, ids, weights, scale):
        writer_ct += accessor(t)
    compute_ct = [
        cb_x,
        cb_w,
        cb_sc,
        cb_act,
        slots_per_core,
        blocks,
        block_k,
        cb_m,
        cb_r,
        cb_one,
    ]
    dm = ttnn.DataMovementConfigDescriptor
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "expert_gate_up_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / f"{KERNELS_Q4}_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / f"{KERNELS_Q4}_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi3,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([w, x, ids, weights, scale, steps, act], program)


def down_q4_op(
    assign,
    cores,
    act,
    ids,
    w,
    steps,
    out,
    *,
    slots,
    kt,
    cols,
    banks,
    bank_cols,
    n_tiles,
    compact_chunk0=None,
):
    """Batch-1 down from Q4_0 tiles (kernels/expert_q4_down_*): as expert_stream.down_op
    with rows=1, plus
    the step tensor steps ([1, E, 32, banks * bank_cols * 32] BF16, interleaved, rows 0
    .. kt - 1 used).
    """
    cb_in0, cb_w, cb_rs, cb_zero, cb_scr, cb_m, cb_racc, cb_one, cb_out = (
        0,
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        16,
    )
    tb = tile_bytes(w.dtype)
    compact = compact_chunk0 is not None
    out_dtype = ttnn.float32 if compact else ttnn.bfloat16
    cbs = [
        cb(cores, cb_in0, ttnn.bfloat16, BF16_TILE, 2 * kt),
        cb(cores, cb_w, w.dtype, tb, 3 * kt * cols),
        cb(cores, cb_rs, ttnn.bfloat16, 64, 1),
        cb(cores, cb_zero, ttnn.bfloat16, 64, 1),
        cb(cores, cb_scr, ttnn.bfloat16, 64, 1),
        cb(cores, cb_m, ttnn.bfloat16, BF16_TILE, 2 * cols),
        cb(cores, cb_racc, ttnn.float32, 2 * BF16_TILE, cols),
        cb(cores, cb_one, ttnn.bfloat16, BF16_TILE, 1),
        cb(cores, cb_out, out_dtype, 2 * BF16_TILE if compact else BF16_TILE, cols),
    ]
    chunks = 2 * n_tiles
    pad = -(-chunks // 64) * 64 - chunks
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
            steps.buffer_address(),
            ids.buffer_address(),
        ]
    id_bytes = _id_bytes(slots)
    reader_ct = [cb_w, cb_rs, slots, kt, cols, bank_cols, tb, id_bytes, *accessor(ids)]
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
        1,
        cb_m,
        cb_one,
        cb_scr,
        id_bytes,
        bank_cols,
    ]
    writer_ct += accessor(act) + accessor(out) + accessor(ids)
    compute_ct = [cb_in0, cb_w, cb_out, slots, kt, cols, cb_m, cb_racc, cb_one]
    dm = ttnn.DataMovementConfigDescriptor
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "expert_down_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "expert_q4_down_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "expert_q4_down_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi3,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([w, act, ids, steps, out], program)


class Q4Experts:
    """One layer's routed experts as Q4_0 tiles and steps, for the batch-1 decode
    streams.

    gate_up and down hold the integers in the layouts of the BFP4 weights
    (expert_stream.py), so the
    weight readers are unchanged; the step tensors follow the writers' layouts."""

    def __init__(
        self, loader, ckpt, layer_idx: int, stream, assemble_gate_up, expert_pad: int
    ):
        p = f"layers.{layer_idx}."
        tp = loader.tp
        memo = {}

        def factored(name):
            if name not in memo:
                if name == "gate_up":  # [E, 2I, H] -> [E, H, 2I], blocks along H
                    memo[name] = q4_factor(
                        ckpt.get(p + "experts.gate_up_proj").transpose(1, 2), dim=-2
                    )
                else:  # [E, H, I] -> [E, I, H], blocks along I
                    memo[name] = q4_factor(
                        ckpt.get(p + "experts.down_proj").transpose(1, 2).contiguous(),
                        dim=-2,
                    )
            return memo[name]

        groups = -(-stream.kx // TILE)
        kt, width = stream.kt, stream.down_width
        self.gate_up = loader.put(
            f"L{layer_idx}.exp_gate_up_q4_p{expert_pad}",
            lambda: assemble_gate_up(factored("gate_up")[0])[None],
            ttnn.bfloat4_b,
            shard_dim=-1,
        )
        self.gate_up_steps = loader.put(
            f"L{layer_idx}.exp_gate_up_q4s_p{expert_pad}",
            lambda: step_tiles(assemble_gate_up(factored("gate_up")[1]), groups)[None],
            ttnn.bfloat16,
            shard_dim=-1,
        )
        self.down = loader.put(
            f"L{layer_idx}.exp_down_q4_p{width}",
            lambda: pad_down(factored("down")[0], width)[None],
            ttnn.bfloat4_b,
            shard_dim=-2,
        )

        def down_steps():  # [E, I / 32, W] -> per chip [E, 32, W], rows 0 .. kt - 1
            s = pad_down(factored("down")[1], width)
            e = s.shape[0]
            s = s.reshape(e, tp, kt, width)
            return torch.nn.functional.pad(s, (0, 0, 0, TILE - kt)).reshape(
                e, tp * TILE, width
            )[None]

        self.down_steps = loader.put(
            f"L{layer_idx}.exp_down_q4s_p{width}",
            down_steps,
            ttnn.bfloat16,
            shard_dim=-2,
        )
        memo.clear()
        self.stream = stream

    def gate_up_op(self, x, ids, weights, scale, act):
        st = self.stream
        return gate_up_q4_op(
            st.gu_assign,
            st.gu_cores,
            x,
            ids,
            weights,
            scale,
            self.gate_up,
            self.gate_up_steps,
            act,
            kx=st.kx,
            slots_per_core=st.gu_slots,
            pairs=st.pairs,
            n_scale=st.n_experts,
        )

    def down_op(self, act, ids, out, compact_chunk0=None):
        st = self.stream
        return down_q4_op(
            st.down_assign,
            st.down_cores,
            act,
            ids,
            self.down,
            self.down_steps,
            out,
            slots=st.slots,
            kt=st.kt,
            cols=st.down_cols,
            banks=st.banks,
            bank_cols=st.bank_cols,
            n_tiles=st.n_tiles,
            compact_chunk0=compact_chunk0,
        )
