// Row form (FP32) -> compact form (BRISC). Core i copies the row-0 chunks of tiles [tile0, tile0 + tiles)
// to compact chunks dst_chunk0 + 2 k + f, and writes `zero_chunks` zero chunks from zero_chunk0.
//
// runtime args: [src_addr, dst_addr, tile0, tiles, dst_chunk0, zero_chunk0, zero_chunks]

#include "compact.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile0 = get_arg_val<uint32_t>(2);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_chunk0 = get_arg_val<uint32_t>(4);
    const uint32_t zero_chunk0 = get_arg_val<uint32_t>(5);
    const uint32_t zero_chunks = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    using RF = RowForm<4>;
    const auto s_src = TensorAccessor(src_args, src_addr, RF::tile_bytes);
    const auto s_dst = TensorAccessor(dst_args, dst_addr, COMPACT_TILE);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t k = 0; k < tiles; ++k) {
        const uint64_t src = s_src.get_noc_addr(tile0 + k);
        noc_async_read(src, scr + (2 * k) * COMPACT_CHUNK, COMPACT_CHUNK);
        noc_async_read(src + RF::face_bytes, scr + (2 * k + 1) * COMPACT_CHUNK, COMPACT_CHUNK);
    }
    const uint32_t zeros = scr + 2 * tiles * COMPACT_CHUNK;
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for (uint32_t i = 0; i < COMPACT_CHUNK / 4; ++i) {
        z[i] = 0;
    }
    noc_async_read_barrier();
    for (uint32_t c = 0; c < 2 * tiles; ++c) {
        noc_async_write(scr + c * COMPACT_CHUNK, compact_chunk_addr(s_dst, dst_chunk0 + 2 * tile0 + c), COMPACT_CHUNK);
    }
    for (uint32_t c = 0; c < zero_chunks; ++c) {
        noc_async_write(zeros, compact_chunk_addr(s_dst, zero_chunk0 + c), COMPACT_CHUNK);
    }
    noc_async_write_barrier();
}
