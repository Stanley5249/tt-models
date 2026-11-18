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

import struct
from pathlib import Path

import torch
import ttnn

KERNELS = Path(__file__).parent / "kernels"
TILE = 32
CHUNK = 16
FP32_TILE = 4096
CHUNKS_PER_TILE = 64
L1 = ttnn.L1_MEMORY_CONFIG


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def compact_tiles(hidden: int) -> int:
    return -(-hidden // CHUNK // CHUNKS_PER_TILE)


def to_compact(v: torch.Tensor) -> torch.Tensor:
    """[n, H] -> [32, n * V * 32] float32 whose TILE layout puts chunk c of vector i at byte 64 c of its V tiles."""
    n, h = v.shape
    tiles = compact_tiles(h)
    chunks = torch.zeros(n, tiles * CHUNKS_PER_TILE, CHUNK)
    chunks[:, : h // CHUNK] = v.float().reshape(n, -1, CHUNK)
    # chunk q of a tile = face q // 16, face row q % 16 -> tile row (face // 2) * 16 + r, cols (face % 2) * 16 + j
    c = chunks.reshape(n, tiles, 2, 2, 16, CHUNK)  # [n, tile, face_row_half, face_col_half, r, j]
    out = c.permute(2, 4, 0, 1, 3, 5).reshape(TILE, n * tiles * TILE)
    return out


def from_compact(t: torch.Tensor, n: int, h: int) -> torch.Tensor:
    """Inverse of to_compact: [32, n * V * 32] -> [n, h]."""
    tiles = compact_tiles(h)
    c = t.float().reshape(2, 16, n, tiles, 2, CHUNK).permute(2, 3, 0, 4, 1, 5)
    return c.reshape(n, tiles * CHUNKS_PER_TILE * CHUNK)[:, :h]


def compact_tensor(mesh, v: torch.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """Host vectors [n, H] -> device compact FP32 [1, 1, 32, n * V * 32], replicated."""
    return ttnn.from_torch(
        to_compact(v)[None, None],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=memory_config,
    )


def _cores(n: int, y0: int = 0) -> list[ttnn.CoreCoord]:
    return [ttnn.CoreCoord(i % 8, y0 + i // 8) for i in range(n)]


def _crs(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(cores, index, dtype, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor(t):
    return ttnn.TensorAccessorArgs(t).get_compile_time_args()


class Boundary:
    """Compact pack and the two tail ops of one mesh (core placement shared by the layers)."""

    def __init__(self, mesh, hidden: int, eps: float, tail_cores: int = 11):
        self.mesh = mesh
        self.hidden = hidden
        self.chunks = hidden // CHUNK  # 176
        self.V = compact_tiles(hidden)  # 3
        self.eps = eps
        self.tail_cores = _cores(tail_cores, y0=0)
        per = -(-self.chunks // tail_cores)
        self.tail_ranges = [(min(i * per, self.chunks), min((i + 1) * per, self.chunks)) for i in range(tail_cores)]

    # -- tensors ---------------------------------------------------------------

    def compact(self, vectors: int = 1) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, vectors * self.V * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, L1
        )

    def row(self) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, L1)

    # -- ops -------------------------------------------------------------------

    def pack(self, src: ttnn.Tensor, dst: ttnn.Tensor, cores: list, dst_chunk0: int = 0) -> ttnn.Tensor:
        """Row form FP32 [1, 1, 1, H] (any layout) -> compact chunks dst_chunk0.. of dst; zeroes the padding."""
        n_tiles = self.hidden // TILE
        per = -(-n_tiles // len(cores))
        crs = _crs(cores)
        rt = ttnn.RuntimeArgs()
        pad0 = dst_chunk0 + self.chunks
        pad = self.V * CHUNKS_PER_TILE - self.chunks
        for i, c in enumerate(cores):
            t0 = min(i * per, n_tiles)
            t1 = min(t0 + per, n_tiles)
            last = i == len(cores) - 1
            rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), t0, t1 - t0, dst_chunk0, pad0, pad if last else 0]
        cb_scr = 0
        scr_bytes = (2 * per + 1) * 64
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "compact_pack.cpp"),
            core_ranges=crs,
            compile_time_args=[cb_scr] + _accessor(src) + _accessor(dst),
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(crs, cb_scr, ttnn.float32, scr_bytes, 1)])
        return ttnn.generic_op([src, dst], program)

    def _tail(self, kind: str, g, x, w, x_out, rows: list, w_vectors: int, scalar: float = 1.0):
        V = self.V
        cores = self.tail_cores
        crs = _crs(cores)
        cb_g, cb_x, cb_w, cb_j, cb_sq, cb_rs, cb_r, cb_t0, cb_xo, cb_row0 = 0, 1, 2, 3, 4, 5, 6, 8, 12, 16
        g_tiles = g.padded_shape[1] * g.padded_shape[3] // TILE
        f32 = ttnn.float32
        cbs = [
            _cb(crs, cb_g, f32, FP32_TILE, g_tiles),
            _cb(crs, cb_x, f32, FP32_TILE, V),
            _cb(crs, cb_w, f32, FP32_TILE, w_vectors * V),
            _cb(crs, cb_j, f32, FP32_TILE, 1),
            _cb(crs, cb_sq, f32, FP32_TILE, V),
            _cb(crs, cb_rs, f32, FP32_TILE, 1),
            _cb(crs, cb_r, f32, FP32_TILE, 1),
            _cb(crs, cb_xo, f32, FP32_TILE, V),
        ]
        cbs += [_cb(crs, cb_t0 + i, f32, FP32_TILE, V) for i in range(4)]
        cbs += [_cb(crs, cb_row0 + i, ttnn.bfloat16, 2048, V) for i in range(len(rows))]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            c0, c1 = self.tail_ranges[i]
            reader_rt[c.x][c.y] = [g.buffer_address(), x.buffer_address(), w.buffer_address()]
            writer_rt[c.x][c.y] = [x_out.buffer_address(), 1 if i == 0 else 0, c0, c1] + [r.buffer_address() for r in rows]
        reader_ct = [cb_g, cb_x, cb_w, cb_j, g_tiles, V, w_vectors * V] + _accessor(g) + _accessor(x) + _accessor(w)
        writer_ct = [cb_xo, cb_row0, V, len(rows)] + _accessor(x_out) + _accessor(rows[0])
        compute_ct = [cb_g, cb_x, cb_w, cb_j, cb_sq, cb_rs, cb_r, cb_t0, cb_xo, cb_row0, V]
        compute_ct += [f32_bits(1.0 / self.hidden), f32_bits(self.eps), f32_bits(scalar), g_tiles // 2, w_vectors * V]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_reader.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_writer.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "tail_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=[(kind, "1")],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([g, x, w, *rows, x_out], program)

    def attn_tail(self, g, x, w, x_out, n, mlp_in, exp_in):
        """g: all-gathered compact attention partials [1, 2, 32, V*32]; x: compact residual; w: compact
        [post_attention | pre_feedforward | pre_feedforward_2] weights. Writes x_out = x + rms(sum g) *
        w_post_attn (compact) and n = rms(x_out), n * w_pre_ff, n * w_pre_ff2 (row form BF16)."""
        return self._tail("ATTN_TAIL", g, x, w, x_out, [n, mlp_in, exp_in], 3)

    def ff_tail(self, g, x, w, x_out, h, scalar: float):
        """g: all-gathered compact [mlp | experts] partials [1, 2, 32, 2*V*32]; w: compact [post_ff1 |
        post_ff2 | post_ff | next input norm]. Writes x_out = (x + rms(rms(m) w1 + rms(e) w2) w_ff) * scalar
        (compact) and h = rms(x_out) * w_in (row form BF16)."""
        return self._tail("FF_TAIL", g, x, w, x_out, [h], 4, scalar)


def rope_partner(width: int, head_dim: int) -> torch.Tensor:
    """Index of every fused-row value's rotate_half partner (the other half of its head)."""
    d = torch.arange(width)
    half = head_dim // 2
    return torch.where(d % head_dim < half, d + half, d - half)


class QkRope:
    """q/k RMSNorm, head weights and RoPE of the fused batch-1 QKV row in one op (kernels/qk_rope_*).

    One core per compact tile (4 heads of 256 or 2 of 512); V heads have weight 1, cos 1, sin 0.
    Replaces reshape, rms_norm, the weight multiply, the rotate_half matmul, three RoPE eltwise ops and the
    reshape back (~60 us per layer)."""

    def __init__(self, mesh, width: int, head_dim: int, eps: float, core_row: int = 6):
        self.mesh, self.width, self.head_dim, self.eps = mesh, width, head_dim, eps
        self.V = width // (CHUNK * CHUNKS_PER_TILE)
        assert self.V * CHUNK * CHUNKS_PER_TILE == width
        self.cores = [ttnn.CoreCoord(i, core_row) for i in range(self.V)]
        self.partner = rope_partner(width, head_dim)
        self.sign = torch.where(torch.arange(width) % head_dim < head_dim // 2, -1.0, 1.0)

    def weights(self, w_row: torch.Tensor) -> ttnn.Tensor:
        """[W] per-value head weights -> compact [w | w permuted to partners]."""
        return compact_tensor(self.mesh, torch.stack([w_row, w_row[self.partner]]))

    def host_tables(self, cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
        """Per-step [cos | sin * rotate_half sign] for one token, as host compact FP32 [1, 1, 32, 2 V 32]."""
        return to_compact(torch.stack([cos_row, sin_row * self.sign]))[None, None]

    def __call__(self, x, cs, w, out):
        """x: FP32 row-form QKV matmul output [1, 1, 1, W]; out: BF16 row form [1, 1, 1, W]."""
        crs = _crs(self.cores)
        cb_x, cb_xr, cb_cs, cb_w, cb_a, cb_b, cb_sq, cb_s1, cb_r, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        f32 = ttnn.float32
        cbs = [_cb(crs, i, f32, FP32_TILE, n) for i, n in ((cb_x, 1), (cb_xr, 1), (cb_cs, 2), (cb_w, 2), (cb_a, 1), (cb_b, 1), (cb_sq, 2), (cb_s1, 1), (cb_r, 1))]
        cbs.append(_cb(crs, cb_out, ttnn.bfloat16, 2048, 1))
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for t, c in enumerate(self.cores):
            reader_rt[c.x][c.y] = [x.buffer_address(), cs.buffer_address(), w.buffer_address(), t]
            writer_rt[c.x][c.y] = [out.buffer_address(), t]
        head_chunks = self.head_dim // CHUNK
        reader_ct = [cb_x, cb_xr, cb_cs, cb_w, cb_a, cb_b, head_chunks, self.V] + _accessor(x) + _accessor(cs) + _accessor(w)
        compute_ct = [cb_x, cb_xr, cb_cs, cb_w, cb_a, cb_b, cb_sq, cb_s1, cb_r, cb_out, f32_bits(1.0 / self.head_dim), f32_bits(self.eps)]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_reader.cpp"), core_ranges=crs, compile_time_args=reader_ct, runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_writer.cpp"), core_ranges=crs, compile_time_args=[cb_out] + _accessor(out),
                runtime_args=writer_rt, config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "qk_rope_compute.cpp"), core_ranges=crs, compile_time_args=compute_ct, runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
            ),
        ]
        return ttnn.generic_op([x, cs, w, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
