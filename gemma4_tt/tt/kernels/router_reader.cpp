// Router input reader (NCRISC): this core's K slice of the normed rows x (rows 0 .. rows - 1 of `kt` BF16
// tiles) and the matching [kt, n_tiles] block of the BF16 router weight. The reducer also builds the sum
// matrix J: J[b, r] = 1 when r % rows == b % rows, so J @ partials adds each token's rows (all ones at
// one row).
//
// runtime args: [x_addr, w_addr, k0, is_reducer]

#include "compact.hpp"
#include "stream_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t k0 = get_arg_val<uint32_t>(2);
    const uint32_t is_reducer = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_j = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t rows = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_w = TensorAccessor(w_args, w_addr, TILE_BYTES_BF16);

    if (is_reducer) {
        cb_reserve_back(cb_j, 1);
        volatile tt_l1_ptr uint32_t* j = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_j));
        for (uint32_t i = 0; i < COMPACT_TILE / 4; ++i) {  // FP32 tile: face i / 256, face row, column
            const uint32_t f = i / 256, row = (f / 2) * 16 + (i % 256) / 16, col = (f % 2) * 16 + i % 16;
            j[i] = col % rows == row % rows ? 0x3F800000 : 0;
        }
        cb_push_back(cb_j, 1);
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t lx = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k0 + k);
        noc_async_read(src, lx + k * TILE_BYTES_BF16, rows * ROW0_FACE_BYTES);
        noc_async_read(src + FACE1_OFFSET, lx + k * TILE_BYTES_BF16 + FACE1_OFFSET, rows * ROW0_FACE_BYTES);
    }
    cb_reserve_back(cb_w, kt * n_tiles);
    const uint32_t lw = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < kt; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            noc_async_read_page((k0 + k) * n_tiles + n, s_w, lw + (k * n_tiles + n) * TILE_BYTES_BF16);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);
    cb_push_back(cb_w, kt * n_tiles);
}
