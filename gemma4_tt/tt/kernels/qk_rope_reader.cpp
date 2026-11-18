// q/k norm + RoPE input gather (NCRISC) for one compact tile of the fused QKV row (boundary.py QkRope).
//
// Gathers compact tile `tile` of the FP32 QKV matmul output (row form) twice: in order (x) and with every
// chunk swapped with its half-head partner (xr, the rotate_half permutation; the sign is in the sin
// table). Reads the step's [cos | sin * sign] and the layer's [w | w permuted] compact tiles, and builds
// the two block-ones tiles of the per-head sums.
//
// runtime args: [x_addr, cs_addr, w_addr, tile]

#include "compact.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cs_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = get_compile_time_arg_val(4);  // S1 = sq @ A
    constexpr uint32_t cb_b = get_compile_time_arg_val(5);  // S2 = B @ S1
    constexpr uint32_t head_chunks = get_compile_time_arg_val(6);  // 16 (head_dim 256) or 32 (512)
    constexpr uint32_t vec_tiles = get_compile_time_arg_val(7);    // compact tiles of the row (cs / w stride)
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto cs_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<cs_args.next_compile_time_args_offset()>();
    using RF = RowForm<4>;

    const auto s_x = TensorAccessor(x_args, x_addr, RF::tile_bytes);
    const auto s_cs = TensorAccessor(cs_args, cs_addr, COMPACT_TILE);
    const auto s_w = TensorAccessor(w_args, w_addr, COMPACT_TILE);

    cb_reserve_back(cb_x, 1);
    cb_reserve_back(cb_xr, 1);
    const uint32_t lx = get_write_ptr(cb_x), lxr = get_write_ptr(cb_xr);
    constexpr uint32_t half = head_chunks / 2;
    for (uint32_t q = 0; q < CHUNKS_PER_TILE; ++q) {
        const uint32_t c = tile * CHUNKS_PER_TILE + q;
        const uint32_t partner = (c % head_chunks) < half ? c + half : c - half;
        noc_async_read(row_chunk_addr<4>(s_x, c), lx + q * COMPACT_CHUNK, COMPACT_CHUNK);
        noc_async_read(row_chunk_addr<4>(s_x, partner), lxr + q * COMPACT_CHUNK, COMPACT_CHUNK);
    }
    cb_reserve_back(cb_cs, 2);
    const uint32_t lcs = get_write_ptr(cb_cs);
    noc_async_read_page(tile, s_cs, lcs);
    noc_async_read_page(vec_tiles + tile, s_cs, lcs + COMPACT_TILE);
    cb_reserve_back(cb_w, 2);
    const uint32_t lw = get_write_ptr(cb_w);
    noc_async_read_page(tile, s_w, lw);
    noc_async_read_page(vec_tiles + tile, s_w, lw + COMPACT_TILE);

    // A: ones on faces 0 and 3 (column halves) for one-face heads, all ones for two-face heads;
    // B: ones on faces 0 and 3 (row halves).
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_a));
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_b));
    for (uint32_t i = 0; i < COMPACT_TILE / 4; ++i) {
        const uint32_t face = i / 256;
        const bool diag = face == 0 || face == 3;
        b[i] = diag ? 0x3F800000 : 0;
        a[i] = (diag || head_chunks == 32) ? 0x3F800000 : 0;
    }
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
    noc_async_read_barrier();
    cb_push_back(cb_x, 1);
    cb_push_back(cb_xr, 1);
    cb_push_back(cb_cs, 2);
    cb_push_back(cb_w, 2);
}
