// Activation gather and output scatter (BRISC, NOC1) of the routed-expert down stream.
//
// 1. Copies the token rows (0 .. rows - 1) of each slot's 11 act tiles (already scaled by the routing
//    weights) into cb_in0, one slot at a time, so the buffer holds two slots whatever the batch.
// 2. For each output column tile n = bank + 12 j < n_tiles: with compact = 0, writes the token rows of the
//    BF16 output tile to tile n of a [32, H] BF16 tensor; with compact = 1 (one row), writes the two FP32
//    chunks of row 0 to compact chunks chunk0 + 2 n + f (compact.hpp), and the core with zero_chunks > 0
//    also writes the vector's zero padding chunks.
//
// runtime args: [act_addr, out_addr, bank, j0, zero_chunk0, zero_chunks]

#include "compact.hpp"
#include "stream_common.hpp"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t zero_chunk0 = get_arg_val<uint32_t>(4);
    const uint32_t zero_chunks = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t banks = get_compile_time_arg_val(4);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t compact = get_compile_time_arg_val(6);
    constexpr uint32_t chunk0 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(8);
    constexpr uint32_t kt = get_compile_time_arg_val(9);  // act tiles per slot
    constexpr uint32_t rows = get_compile_time_arg_val(10);
    constexpr auto act_args = TensorAccessorArgs<11>();
    constexpr uint32_t row_bytes = rows * ROW0_FACE_BYTES;
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    const auto s_act = TensorAccessor(act_args, act_addr, TILE_BYTES_BF16);
    const auto s_out = TensorAccessor(out_args, out_addr, compact ? COMPACT_TILE : TILE_BYTES_BF16);

    for (uint32_t e = 0; e < slots; ++e) {
        cb_reserve_back(cb_in0, kt);
        const uint32_t in0 = get_write_ptr(cb_in0);
        for (uint32_t k = 0; k < kt; ++k) {
            const uint64_t src = s_act.get_noc_addr(e * kt + k);
            const uint32_t dst = in0 + k * TILE_BYTES_BF16;
            noc_async_read(src, dst, row_bytes);
            noc_async_read(src + FACE1_OFFSET, dst + FACE1_OFFSET, row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, kt);
    }

    if constexpr (compact) {
        if (zero_chunks) {
            cb_reserve_back(cb_zero, 1);
            const uint32_t z = get_write_ptr(cb_zero);
            volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
            for (uint32_t i = 0; i < COMPACT_CHUNK / 4; ++i) {
                zp[i] = 0;
            }
            for (uint32_t c = 0; c < zero_chunks; ++c) {
                noc_async_write(z, compact_chunk_addr(s_out, zero_chunk0 + c), COMPACT_CHUNK);
            }
        }
    }
    for (uint32_t c = 0; c < cols; ++c) {
        cb_wait_front(cb_out, 1);
        const uint32_t n = bank + banks * (j0 + c);
        if (n < n_tiles) {
            const uint32_t src = get_read_ptr(cb_out);
            if constexpr (compact) {
                using RF = RowForm<4>;
                noc_async_write(src, compact_chunk_addr(s_out, chunk0 + 2 * n), COMPACT_CHUNK);
                noc_async_write(src + RF::face_bytes, compact_chunk_addr(s_out, chunk0 + 2 * n + 1), COMPACT_CHUNK);
            } else {
                const uint64_t dst = s_out.get_noc_addr(n);
                noc_async_write(src, dst, row_bytes);
                noc_async_write(src + FACE1_OFFSET, dst + FACE1_OFFSET, row_bytes);
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
