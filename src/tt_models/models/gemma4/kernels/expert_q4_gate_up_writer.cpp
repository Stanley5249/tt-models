// Activation rows, step tiles and output scatter (BRISC, NOC1) of the Q4_0 gate|up stream.
//
// - cb_x tile k holds row `row` of x's K tile k in its row k % 32 and zeros elsewhere
//   (expert_q4_gate_up_compute.cpp). The tiles are zeroed by doubling local copies of one zero tile,
//   then filled and pushed group by group (32 K tiles), so compute can start on the first group.
// - cb_m: each slot's 2 * groups step tiles, read at the start from this core's DRAM bank. The step
//   tensor is interleaved BF16 [E * groups * 32, 24 * 32]; bank b holds columns (gate b, up 12 + b) of
//   every (expert, group) row of tiles, so one expert's tiles are contiguous there.
// - cb_one: a ones row (row 0) for the group sums.
// - cb_sc: two score tiles per slot (routing weight, per_expert_scale), as expert_gate_up_writer.cpp.
//
// row: the token's row of x and act (this core's slots slot0 .. slot0 + slots - 1 are all its slots).
// runtime args: [x_addr, act_addr, idx_addr, wts_addr, scale_addr, pair, slot0, m_addr, bank, row]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t wts_addr = get_arg_val<uint32_t>(3);
    const uint32_t scale_addr = get_arg_val<uint32_t>(4);
    const uint32_t pair = get_arg_val<uint32_t>(5);
    const uint32_t slot0 = get_arg_val<uint32_t>(6);
    const uint32_t m_addr = get_arg_val<uint32_t>(7);
    const uint32_t bank = get_arg_val<uint32_t>(8);
    const uint32_t row = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t kx = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t pairs = get_compile_time_arg_val(6);
    constexpr uint32_t num_experts = get_compile_time_arg_val(7);
    constexpr uint32_t id_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_m = get_compile_time_arg_val(9);
    constexpr uint32_t cb_one = get_compile_time_arg_val(10);
    constexpr uint32_t rows = get_compile_time_arg_val(11);  // token rows of x; the act row is `row`
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto wts_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();
    constexpr uint32_t groups = (kx + 31) / 32;
    constexpr uint32_t m_expert_bytes = 2 * groups * TILE_BYTES_BF16;  // one expert's step tiles in a bank

    const auto s_x = TensorAccessor(x_args, x_addr, TILE_BYTES_BF16);
    const auto s_act = TensorAccessor(act_args, act_addr, TILE_BYTES_BF16);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, id_bytes);
    const auto s_wts = TensorAccessor(wts_args, wts_addr, id_bytes);
    const auto s_scale = TensorAccessor(scale_args, scale_addr, num_experts * 2);

    // scratch: ids at 0, weights at 64, scale table at 128
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    read_row0(s_idx, scr, id_bytes);
    read_row0(s_wts, scr + 64, id_bytes);
    noc_async_read(s_scale.get_noc_addr(0), scr + 128, num_experts * 2);

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    noc_async_read_barrier();  // ids
    // Slot 0's steps now; the others once x is in place, so they do not delay the weight stream's start.
    cb_reserve_back(cb_m, 2 * groups * slots);
    const uint64_t m_bank = get_noc_addr_from_bank_id<true>(bank, m_addr);
    const uint32_t m_l1 = get_write_ptr(cb_m);
    auto read_steps = [&](uint32_t i) {
        const uint32_t src = static_cast<uint32_t>(ids[slot0 + i]) * m_expert_bytes;
        noc_async_read(m_bank + src, m_l1 + i * m_expert_bytes, m_expert_bytes);
    };
    read_steps(0);

    // Zero cb_x: one tile from MEM_ZEROS, then doubling local copies.
    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < TILE_BYTES_BF16; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    for (uint32_t done = TILE_BYTES_BF16; done < kx * TILE_BYTES_BF16;) {
        const uint32_t n = done < kx * TILE_BYTES_BF16 - done ? done : kx * TILE_BYTES_BF16 - done;
        noc_async_read(get_noc_addr(x_l1), x_l1 + done, n);
        noc_async_read_barrier();
        done += n;
    }

    const uint32_t src_row = (row % 16) * ROW0_FACE_BYTES + (row / 16) * 2 * FACE1_OFFSET;
    static_assert(rows <= 32, "x has at most 32 token rows");
    for (uint32_t k0 = 0; k0 < kx; k0 += 32) {
        const uint32_t nk = kx - k0 < 32 ? kx - k0 : 32;
        for (uint32_t r = 0; r < nk; ++r) {
            const uint64_t src = s_x.get_noc_addr(k0 + r) + src_row;
            const uint32_t dst = x_l1 + (k0 + r) * TILE_BYTES_BF16 + (r / 16) * 2 * FACE1_OFFSET + (r % 16) * ROW0_FACE_BYTES;
            noc_async_read(src, dst, ROW0_FACE_BYTES);
            noc_async_read(src + FACE1_OFFSET, dst + FACE1_OFFSET, ROW0_FACE_BYTES);
        }
        noc_async_read_barrier();
        if (k0 == 0) {
            cb_push_back(cb_m, 2 * groups);
        }
        cb_push_back(cb_x, nk);
    }
    for (uint32_t i = 1; i < slots; ++i) {
        read_steps(i);
        noc_async_read_barrier();
        cb_push_back(cb_m, 2 * groups);
    }

    cb_reserve_back(cb_one, 1);
    {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_one));
        for (uint32_t i = 0; i < TILE_BYTES_BF16 / 4; ++i) {
            t[i] = 0;
        }
        for (uint32_t c = 0; c < 8; ++c) {  // row 0 of faces 0 and 1: 16 BF16 ones each
            t[c] = 0x3F803F80;
            t[FACE1_OFFSET / 4 + c] = 0x3F803F80;
        }
    }
    cb_push_back(cb_one, 1);

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

    for (uint32_t i = 0; i < slots; ++i) {
        cb_wait_front(cb_act, 1);
        const uint32_t src = get_read_ptr(cb_act);
        const uint64_t dst = s_act.get_noc_addr((slot0 + i) * pairs + pair) + src_row;
        noc_async_write(src, dst, ROW0_FACE_BYTES);
        noc_async_write(src + FACE1_OFFSET, dst + FACE1_OFFSET, ROW0_FACE_BYTES);
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
}
