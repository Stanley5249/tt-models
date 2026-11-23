// Activation gather and output scatter (BRISC, NOC1) of the routed-expert gate|up stream.
//
// Token rows 0 .. rows - 1 are the decode tokens. The ids row holds the step's distinct routed experts
// (one slot each) and their count at entry count_index; this core takes the contiguous run of slots
// slot_run(count, readers, rank) (stream_common.hpp), as its reader does. With row_weights = 1 the
// weights row is slot-major [slot * rows + row]: a token's routing weight for that slot's expert, zero
// when the token did not pick it. With row_weights = 0 (the dense MLP, whose "slots" are column groups)
// entry `slot` serves every row.
//
// 1. Builds two score tiles per slot of the run: the token rows' routing weights, and
//    per_expert_scale[id] (BF16). Compute multiplies the activation by both, so a slot only contributes
//    to the rows of the tokens that routed to it.
// 2. Copies the token rows of the 88 x tiles (the rows below them stay garbage and only produce
//    garbage rows) into cb_x: one read of rows * 32 bytes per face.
// 3. Pushes the run's length to cb_cnt for compute.
// 4. Writes the token rows of every output tile to tile (slot * pairs + pair) of the act tensor.
//
// runtime args: [x_addr, act_addr, idx_addr, wts_addr, scale_addr, pair, rank]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t wts_addr = get_arg_val<uint32_t>(3);
    const uint32_t scale_addr = get_arg_val<uint32_t>(4);
    const uint32_t pair = get_arg_val<uint32_t>(5);
    const uint32_t rank = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t kx = get_compile_time_arg_val(4);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(5);
    constexpr uint32_t pairs = get_compile_time_arg_val(6);
    constexpr uint32_t num_experts = get_compile_time_arg_val(7);
    constexpr uint32_t rows = get_compile_time_arg_val(8);
    constexpr uint32_t row_weights = get_compile_time_arg_val(9);
    constexpr uint32_t id_bytes = get_compile_time_arg_val(10);   // bytes read of the ids row
    constexpr uint32_t wts_bytes = get_compile_time_arg_val(11);  // bytes read of the weights row
    constexpr uint32_t count_index = get_compile_time_arg_val(12);
    constexpr uint32_t readers = get_compile_time_arg_val(13);
    constexpr auto x_args = TensorAccessorArgs<14>();
    constexpr uint32_t row_bytes = rows * ROW0_FACE_BYTES;
    static_assert(row_bytes <= FACE1_OFFSET, "token rows must fit the top faces (16 rows)");
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto wts_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_act = TensorAccessor(act_args, act_addr, TILE_BYTES_BF16);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, id_bytes);
    const auto s_wts = TensorAccessor(wts_args, wts_addr, wts_bytes);
    const auto s_scale = TensorAccessor(scale_args, scale_addr, num_experts * 2);

    // scratch: ids, then weights, then the scale table
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    read_row0(s_idx, scr, id_bytes);
    read_row0(s_wts, scr + id_bytes, wts_bytes);
    noc_async_read(s_scale.get_noc_addr(0), scr + id_bytes + wts_bytes, num_experts * 2);

    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kx; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        const uint32_t dst = x_l1 + k * TILE_BYTES_BF16;
        noc_async_read(src, dst, row_bytes);
        noc_async_read(src + FACE1_OFFSET, dst + FACE1_OFFSET, row_bytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + id_bytes);
    volatile tt_l1_ptr uint16_t* scale =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + id_bytes + wts_bytes);
    const SlotRun run = slot_run(ids[count_index], readers, rank);
    const uint32_t m = run.hi - run.lo;

    cb_reserve_back(cb_cnt, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt)) = m;
    cb_push_back(cb_cnt, 1);

    cb_reserve_back(cb_sc, 2 * m);
    const uint32_t sc_l1 = get_write_ptr(cb_sc);
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t s = run.lo + i;
        volatile tt_l1_ptr uint16_t* tw =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_l1 + 2 * i * TILE_BYTES_BF16);
        volatile tt_l1_ptr uint16_t* ts =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sc_l1 + (2 * i + 1) * TILE_BYTES_BF16);
        const uint16_t sv = scale[ids[s]];
        for (uint32_t r = 0; r < rows; ++r) {
            const uint16_t wv = row_weights ? wts[s * rows + r] : wts[s];
            for (uint32_t c = 0; c < 16; ++c) {
                tw[r * 16 + c] = wv;
                tw[FACE1_OFFSET / 2 + r * 16 + c] = wv;
                ts[r * 16 + c] = sv;
                ts[FACE1_OFFSET / 2 + r * 16 + c] = sv;
            }
        }
    }
    cb_push_back(cb_sc, 2 * m);
    cb_push_back(cb_x, kx);

    for (uint32_t i = 0; i < m; ++i) {
        cb_wait_front(cb_act, 1);
        const uint32_t src = get_read_ptr(cb_act);
        const uint64_t dst = s_act.get_noc_addr((run.lo + i) * pairs + pair);
        noc_async_write(src, dst, row_bytes);
        noc_async_write(src + FACE1_OFFSET, dst + FACE1_OFFSET, row_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
}
