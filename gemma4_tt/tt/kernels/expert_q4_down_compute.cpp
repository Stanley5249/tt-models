// Routed-expert down for Q4_0 weights (TRISC), batch-1 decode.
//
// As kernels/expert_q4_gate_up_compute.cpp: the weight tiles hold the Q4_0 integers, act K tile k
// (the expert's intermediate block k, k < kt <= 32) sits in row k of an otherwise zero cb_in0 tile, so
// one slot's K tiles leave P[k, n] = act_k . q_k[:, n] in separate rows of a dest tile, and
//     out[n] = sum over slots e and rows k of s_e[k, n] * P_e[k, n].
// Per slot and column the SFPU multiplies P by the step tile (cb_m), and the packer accumulates the
// products of all slots in L1 (FP32, cb_racc). One matmul of a ones row (cb_one) per column then sums
// the rows. Columns run in pairs: two P and two step tiles fill the four FP32 dest tiles.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t slots = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t cb_m = get_compile_time_arg_val(6);
    constexpr uint32_t cb_racc = get_compile_time_arg_val(7);
    constexpr uint32_t cb_one = get_compile_time_arg_val(8);
    static_assert(cols % 2 == 0 && kt <= 32, "column pairs; one group of K blocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);
    cb_reserve_back(cb_racc, cols);
    for (uint32_t e = 0; e < slots; ++e) {
        cb_wait_front(cb_in0, kt);
        cb_wait_front(cb_w, kt * cols);
        cb_wait_front(cb_m, cols);
        for (uint32_t c = 0; c < cols; c += 2) {
            reconfig_data_format_srca(cb_w);
            matmul_init(cb_in0, cb_w);
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_tiles(cb_in0, cb_w, k, k * cols + c, 0);
                matmul_tiles(cb_in0, cb_w, k, k * cols + c + 1, 1);
            }
            reconfig_data_format_srca(cb_w, cb_m);
            copy_init(cb_m);
            copy_tile(cb_m, c, 2);
            copy_tile(cb_m, c + 1, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 3, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_racc);
            pack_reconfig_l1_acc(e > 0);
            pack_tile<true>(0, cb_racc, c);
            pack_tile<true>(1, cb_racc, c + 1);
            tile_regs_release();
        }
        cb_pop_front(cb_m, cols);
        cb_pop_front(cb_w, kt * cols);
        cb_pop_front(cb_in0, kt);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_racc, cols);

    cb_wait_front(cb_racc, cols);
    cb_wait_front(cb_one, 1);
    reconfig_data_format_srca(cb_racc);
    matmul_init(cb_one, cb_racc);
    tile_regs_acquire();
    for (uint32_t c = 0; c < cols; ++c) {
        matmul_tiles(cb_one, cb_racc, 0, c, c);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
    cb_pop_front(cb_racc, cols);
    cb_pop_front(cb_one, 1);
}
