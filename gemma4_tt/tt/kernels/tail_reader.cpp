// Layer-boundary input reader (NCRISC): the all-gathered compact partial sums (both chips), the compact
// residual and the compact weights, whole FP32 tiles, plus an all-ones FP32 tile for the matmul sums.
//
// runtime args: [g_addr, x_addr, w_addr]

#include "compact.hpp"

void kernel_main() {
    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_j = get_compile_time_arg_val(3);
    constexpr uint32_t g_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(6);
    constexpr auto g_args = TensorAccessorArgs<7>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const auto s_g = TensorAccessor(g_args, g_addr, COMPACT_TILE);
    const auto s_x = TensorAccessor(x_args, x_addr, COMPACT_TILE);
    const auto s_w = TensorAccessor(w_args, w_addr, COMPACT_TILE);

    cb_reserve_back(cb_j, 1);
    volatile tt_l1_ptr uint32_t* j = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_j));
    for (uint32_t i = 0; i < COMPACT_TILE / 4; ++i) {
        j[i] = 0x3F800000;  // 1.0f
    }
    cb_push_back(cb_j, 1);

    cb_reserve_back(cb_w, w_tiles);
    uint32_t l1 = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < w_tiles; ++t) {
        noc_async_read_page(t, s_w, l1 + t * COMPACT_TILE);
    }
    cb_reserve_back(cb_x, x_tiles);
    l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < x_tiles; ++t) {
        noc_async_read_page(t, s_x, l1 + t * COMPACT_TILE);
    }
    cb_reserve_back(cb_g, g_tiles);
    l1 = get_write_ptr(cb_g);
    for (uint32_t t = 0; t < g_tiles; ++t) {
        noc_async_read_page(t, s_g, l1 + t * COMPACT_TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, w_tiles);
    cb_push_back(cb_x, x_tiles);
    cb_push_back(cb_g, g_tiles);
}
