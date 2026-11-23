// q/k norm + RoPE output scatter (BRISC): for each token row of this core, the BF16 compact tile `tile`
// back to that row of the fused QKV rows (row form [32, W] BF16), 64 chunks of 32 bytes.
//
// runtime args: [out_addr, tile, n_rows, row 0, row 1, ...]

#include "compact.hpp"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile = get_arg_val<uint32_t>(1);
    const uint32_t n_rows = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    using RF = RowForm<2>;
    const auto s_out = TensorAccessor(out_args, out_addr, RF::tile_bytes);

    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t row = get_arg_val<uint32_t>(3 + i);
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t q = 0; q < CHUNKS_PER_TILE; ++q) {
            noc_async_write(
                src + q * RF::chunk_bytes, row_chunk_addr<2>(s_out, tile * CHUNKS_PER_TILE + q, row), RF::chunk_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
