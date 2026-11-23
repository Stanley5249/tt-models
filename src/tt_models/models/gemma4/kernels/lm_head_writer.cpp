// LM head activation gather and logit scatter (BRISC, NOC1).
//
// 1. Copies the token rows (0 .. rows - 1, rows <= 32) of the kt BF16 x tiles into cb_in0.
// 2. For each FP32 output tile of this core's groups, column tile n = bank + banks * j, writes each token
//    row's 32 logits (two 64-byte face rows) to row `row` of the row-major FP32 [rows, n_tiles * 32]
//    output at byte n * 128. Padding columns (n >= n_tiles) are dropped.
//
// runtime args: [x_addr, out_addr, bank, j0, groups]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t groups = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t group = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(rows <= 32, "token rows fit one tile");
    constexpr uint32_t top = rows < 16 ? rows : 16;  // rows in faces 0 / 1
    constexpr uint32_t bottom = rows - top;            // rows in faces 2 / 3
    constexpr uint32_t f32_face = 1024;

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_out = TensorAccessor(out_args, out_addr, n_tiles * 32 * 4);

    cb_reserve_back(cb_in0, kt);
    const uint32_t x_l1 = get_write_ptr(cb_in0);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        const uint32_t dst = x_l1 + k * TILE_BYTES_BF16;
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(src + f * FACE1_OFFSET, dst + f * FACE1_OFFSET, top * ROW0_FACE_BYTES);
            if constexpr (bottom > 0) {
                noc_async_read(src + 2 * FACE1_OFFSET + f * FACE1_OFFSET,
                               dst + 2 * FACE1_OFFSET + f * FACE1_OFFSET, bottom * ROW0_FACE_BYTES);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, kt);

    for (uint32_t g = 0; g < groups; ++g) {
        cb_wait_front(cb_out, group);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < group; ++c) {
            const uint32_t n = bank + banks * (j0 + g * group + c);
            if (n >= n_tiles) {
                continue;
            }
            const uint32_t tile = src + c * 4 * f32_face;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t face0 = (r < 16 ? 0 : 2 * f32_face) + (r % 16) * 64;
                const uint64_t dst = s_out.get_noc_addr(r, n * 128);
                noc_async_write(tile + face0, dst, 64);
                noc_async_write(tile + face0 + f32_face, dst + 64, 64);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, group);
    }
    noc_async_write_barrier();
}
