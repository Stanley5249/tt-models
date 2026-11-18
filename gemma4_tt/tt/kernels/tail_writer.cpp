// Layer-boundary output writer (BRISC). In the order compute packs them:
// 1. the compact FP32 residual (V tiles), written by the core with write_compact = 1;
// 2. n_row BF16 vectors in compact order, of which this core scatters chunks [c0, c1) into the row form
//    (row 0 of [32, H] BF16 tiles) of output i.
//
// runtime args: [x_out_addr, write_compact, c0, c1, row_addr_0, ..., row_addr_{n_row-1}]

#include "compact.hpp"

void kernel_main() {
    const uint32_t x_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t write_compact = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t c1 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_xo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_row0 = get_compile_time_arg_val(1);  // row outputs use cb_row0, cb_row0 + 1, ...
    constexpr uint32_t V = get_compile_time_arg_val(2);
    constexpr uint32_t n_row = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto row_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();  // all row outputs alike
    using RF = RowForm<2>;
    constexpr uint32_t bf16_chunk = RF::chunk_bytes;

    const auto s_x = TensorAccessor(x_args, x_out_addr, COMPACT_TILE);
    cb_wait_front(cb_xo, V);
    if (write_compact) {
        const uint32_t src = get_read_ptr(cb_xo);
        for (uint32_t t = 0; t < V; ++t) {
            noc_async_write_page(t, s_x, src + t * COMPACT_TILE);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_xo, V);

    for (uint32_t i = 0; i < n_row; ++i) {
        const auto s_row = TensorAccessor(row_args, get_arg_val<uint32_t>(4 + i), RF::tile_bytes);
        const uint32_t cb = cb_row0 + i;
        cb_wait_front(cb, V);
        const uint32_t src = get_read_ptr(cb);
        for (uint32_t c = c0; c < c1; ++c) {
            noc_async_write(src + c * bf16_chunk, row_chunk_addr<2>(s_row, c), bf16_chunk);
        }
        noc_async_write_barrier();
        cb_pop_front(cb, V);
    }
}
