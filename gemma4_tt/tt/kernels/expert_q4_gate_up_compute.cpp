// Routed-expert gate|up for Q4_0 weights (TRISC), one decode token per core.
//
// The weight tiles hold the Q4_0 integers q (BFP4 holds -7..7 exactly); cb_m holds the steps
// s[block, col], one BF16 tile per (slot, group of 32 input blocks, gate|up), row r = block 32 g + r.
// W[k, n] = q[k, n] * s[k / 32, n] and a 32-input block is one K tile, so cb_x tile k holds x's K tile
// k in its row k % 32 and zeros elsewhere (expert_q4_gate_up_writer.cpp). The K tiles of a group then
// accumulate their per-block partial sums P[r, n] in separate rows of one dest tile, and
//     y[n] = sum over groups g and rows r of s[r, n] * P[r, n]
// is an SFPU multiply by the step tile (R = s * P, to L1 as FP32) and, per group, one matmul of a ones
// row (cb_one) with R. The kernel runs at HiFi3, so the sum keeps R at TF32 precision; the weights are
// streaming-bound, so the fidelity costs no time.
//
// After the sum, as kernels/expert_gate_up_compute.cpp: act = gelu_tanh(g) * u * w_e * per_expert_scale.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);
    constexpr uint32_t blocks = get_compile_time_arg_val(5);
    constexpr uint32_t ksteps = get_compile_time_arg_val(6);  // k-steps per weight block (2 tiles each)
    constexpr uint32_t cb_m = get_compile_time_arg_val(7);
    constexpr uint32_t cb_r = get_compile_time_arg_val(8);
    constexpr uint32_t cb_one = get_compile_time_arg_val(9);
    constexpr uint32_t kx = blocks * ksteps;
    constexpr uint32_t groups = (kx + 31) / 32;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_act);
    for (uint32_t i = 0; i < slots; ++i) {
        cb_wait_front(cb_m, 2 * groups);
        uint32_t k = 0;
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t kend = k + 32 < kx ? k + 32 : kx;
            if (i == 0) {
                cb_wait_front(cb_x, kend);  // the writer pushes x group by group
            }
            reconfig_data_format_srca(cb_w);
            matmul_init(cb_x, cb_w);
            tile_regs_acquire();
            for (; k < kend; ++k) {
                const uint32_t t = k % ksteps;
                if (t == 0) {
                    cb_wait_front(cb_w, 2 * ksteps);
                }
                matmul_tiles(cb_x, cb_w, k, 2 * t, 0);
                matmul_tiles(cb_x, cb_w, k, 2 * t + 1, 1);
                if (t == ksteps - 1) {
                    cb_pop_front(cb_w, 2 * ksteps);
                }
            }
            reconfig_data_format_srca(cb_w, cb_m);
            copy_init(cb_m);
            copy_tile(cb_m, 2 * g, 2);
            copy_tile(cb_m, 2 * g + 1, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 3, 1);
            tile_regs_commit();
            cb_reserve_back(cb_r, 2);
            tile_regs_wait();
            pack_reconfig_data_format(cb_r);
            pack_tile(0, cb_r);
            pack_tile(1, cb_r);
            tile_regs_release();
            cb_push_back(cb_r, 2);
        }
        cb_pop_front(cb_m, 2 * groups);

        if (i == 0) {
            cb_wait_front(cb_one, 1);
            cb_wait_front(cb_sc, 2 * slots);
        }
        cb_wait_front(cb_r, 2 * groups);
        reconfig_data_format_srca(cb_r);
        matmul_init(cb_one, cb_r);
        tile_regs_acquire();
        for (uint32_t g = 0; g < groups; ++g) {
            matmul_tiles(cb_one, cb_r, 0, 2 * g, 0);
            matmul_tiles(cb_one, cb_r, 0, 2 * g + 1, 1);
        }
        cb_pop_front(cb_r, 2 * groups);
        reconfig_data_format_srca(cb_r, cb_sc);
        copy_init(cb_sc);
        copy_tile(cb_sc, 2 * i, 2);
        copy_tile(cb_sc, 2 * i + 1, 3);
        gelu_tanh_tile_init();
        gelu_tanh_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(0, 3, 0);
        tile_regs_commit();
        cb_reserve_back(cb_act, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_act);
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);
    }
    cb_pop_front(cb_sc, 2 * slots);
    cb_pop_front(cb_x, kx);
    cb_pop_front(cb_one, 1);
}
