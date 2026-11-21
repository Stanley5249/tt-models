// Layer-boundary input reader (NCRISC): this core's token row of the all-gathered compact partial sums
// and of the compact residual, the compact weights, whole FP32 tiles, plus the reduce scaler tile. The
// gathered tensor holds per chip `vecs` blocks of `rows` vectors (the feed-forward's [mlp rows | expert
// rows]); this row's vectors are packed as [chip 0: vec 0 .. | chip 1: vec 0 ..], so compute sees the
// batch-1 layout (tail_compute.cpp).
//
// runtime args: [g_addr, x_addr, w_addr, x_page0, row] (x_page0: the residual's first tile in a combined input)

#include "compact.hpp"

void kernel_main() {
    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t x_page0 = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_j = get_compile_time_arg_val(3);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t V = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t vecs = get_compile_time_arg_val(7);  // gathered vectors per row: 1 (attention) or 2
    constexpr auto g_args = TensorAccessorArgs<8>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const auto s_g = TensorAccessor(g_args, g_addr, COMPACT_TILE);
    const auto s_x = TensorAccessor(x_args, x_addr, COMPACT_TILE);
    const auto s_w = TensorAccessor(w_args, w_addr, COMPACT_TILE);

    cb_reserve_back(cb_j, 1);
    volatile tt_l1_ptr uint32_t* j = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_j));
    for (uint32_t i = 0; i < COMPACT_TILE / 4; ++i) {
        j[i] = (i % 256) < 16 ? 0x3F800000 : 0;  // reduce scaler: 1.0f in the first row of every face
    }
    cb_push_back(cb_j, 1);

    // The partials and the residual first: compute starts on them while the weights arrive.
    cb_reserve_back(cb_g, 2 * vecs * V);
    uint32_t l1 = get_write_ptr(cb_g);
    for (uint32_t chip = 0; chip < 2; ++chip) {
        for (uint32_t v = 0; v < vecs; ++v) {
            const uint32_t src = chip * vecs * rows * V + (v * rows + row) * V;  // page in the gathered tensor
            const uint32_t dst = (chip * vecs + v) * V;                          // slot in cb_g
            for (uint32_t t = 0; t < V; ++t) {
                noc_async_read_page(src + t, s_g, l1 + (dst + t) * COMPACT_TILE);
            }
        }
    }
    cb_reserve_back(cb_x, V);
    l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < V; ++t) {
        noc_async_read_page(x_page0 + row * V + t, s_x, l1 + t * COMPACT_TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_g, 2 * vecs * V);
    cb_push_back(cb_x, V);
    cb_reserve_back(cb_w, w_tiles);
    l1 = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < w_tiles; ++t) {
        noc_async_read_page(t, s_w, l1 + t * COMPACT_TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, w_tiles);
}
