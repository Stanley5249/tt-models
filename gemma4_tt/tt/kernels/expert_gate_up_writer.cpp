// Activation gather and output scatter (BRISC, NOC1) of the routed-expert gate|up stream.
//
// 1. Builds two score tiles per slot this core owns, row 0 filled with the routing weight w_e and with
//    per_expert_scale[id_e] (BF16); compute multiplies the activation by both.
// 2. Copies row 0 of the 88 x tiles (the decode token; the other rows stay garbage and only produce
//    garbage rows) into cb_x: two 32-byte reads per tile.
// 3. Writes row 0 of every output tile to tile (slot * pairs + pair) of the [32, 8 * 352] act tensor.
//
// runtime args: [x_addr, act_addr, idx_addr, wts_addr, scale_addr, pair, slot0]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t wts_addr = get_arg_val<uint32_t>(3);
    const uint32_t scale_addr = get_arg_val<uint32_t>(4);
    const uint32_t pair = get_arg_val<uint32_t>(5);
    const uint32_t slot0 = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t kx = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t pairs = get_compile_time_arg_val(6);
    constexpr uint32_t num_experts = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto wts_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_act = TensorAccessor(act_args, act_addr, TILE_BYTES_BF16);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, 32);
    const auto s_wts = TensorAccessor(wts_args, wts_addr, 32);
    const auto s_scale = TensorAccessor(scale_args, scale_addr, num_experts * 2);

    // scratch: ids at 0, weights at 64, scale table at 128
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    read_row0(s_idx, scr, 32);
    read_row0(s_wts, scr + 64, 32);
    noc_async_read(s_scale.get_noc_addr(0), scr + 128, num_experts * 2);

    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kx; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        const uint32_t dst = x_l1 + k * TILE_BYTES_BF16;
        noc_async_read(src, dst, ROW0_FACE_BYTES);
        noc_async_read(src + FACE1_OFFSET, dst + FACE1_OFFSET, ROW0_FACE_BYTES);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 64);
    volatile tt_l1_ptr uint16_t* scale = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 128);
    cb_reserve_back(cb_sc, 2 * slots);
    const uint32_t sc_l1 = get_write_ptr(cb_sc);
    for (uint32_t i = 0; i < slots; ++i) {
        const uint16_t vals[2] = {wts[slot0 + i], scale[ids[slot0 + i]]};
        for (uint32_t j = 0; j < 2; ++j) {
            volatile tt_l1_ptr uint16_t* t =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_l1 + (2 * i + j) * TILE_BYTES_BF16);
            for (uint32_t c = 0; c < 16; ++c) {
                t[c] = vals[j];
                t[FACE1_OFFSET / 2 + c] = vals[j];
            }
        }
    }
    cb_push_back(cb_sc, 2 * slots);
    cb_push_back(cb_x, kx);

    for (uint32_t i = 0; i < slots; ++i) {
        cb_wait_front(cb_act, 1);
        const uint32_t src = get_read_ptr(cb_act);
        const uint64_t dst = s_act.get_noc_addr((slot0 + i) * pairs + pair);
        noc_async_write(src, dst, ROW0_FACE_BYTES);
        noc_async_write(src + FACE1_OFFSET, dst + FACE1_OFFSET, ROW0_FACE_BYTES);
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
}
