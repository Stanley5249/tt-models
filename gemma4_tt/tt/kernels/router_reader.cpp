// Router input reader (NCRISC): this core's K slice of the normed row x (row 0 of `kt` BF16 tiles) and
// the matching [kt, n_tiles] block of the BF16 router weight; the reducer also builds an all-ones tile.
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
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_w = TensorAccessor(w_args, w_addr, TILE_BYTES_BF16);

    if (is_reducer) {
        cb_reserve_back(cb_j, 1);
        volatile tt_l1_ptr uint32_t* j = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_j));
        for (uint32_t i = 0; i < COMPACT_TILE / 4; ++i) {
            j[i] = 0x3F800000;
        }
        cb_push_back(cb_j, 1);
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t lx = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k0 + k);
        noc_async_read(src, lx + k * TILE_BYTES_BF16, ROW0_FACE_BYTES);
        noc_async_read(src + FACE1_OFFSET, lx + k * TILE_BYTES_BF16 + FACE1_OFFSET, ROW0_FACE_BYTES);
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
