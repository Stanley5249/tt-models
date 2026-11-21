// Routed-expert gate|up matmul + GELU-tanh (TRISC) for Gemma 4 batch-1 decode.
//
// For each slot this core owns: g = x . W[:, gate b], u = x . W[:, up b] over 88 K tiles (FP32 dest),
// act = gelu_tanh(g) * u * w_e * per_expert_scale[e], packed as one BF16 tile (row 0 is the token).

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
    constexpr uint32_t ksteps = get_compile_time_arg_val(6);  // k-steps per block (2 tiles each)
    constexpr uint32_t kx = blocks * ksteps;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_act);
    cb_wait_front(cb_x, kx);
    cb_wait_front(cb_sc, 2 * slots);
    // Matmul unpacks in1 (BFP4 weights) to SrcA; the score copies need SrcA as BF16. Without the
    // compute-kernel sentinel nothing reconfigures the format on its own, so switch it explicitly.
    for (uint32_t i = 0; i < slots; ++i) {
        if (i) {
            reconfig_data_format_srca(cb_sc, cb_w);
        }
        matmul_init(cb_x, cb_w);
        tile_regs_acquire();
        for (uint32_t b = 0; b < blocks; ++b) {
            cb_wait_front(cb_w, 2 * ksteps);
            for (uint32_t t = 0; t < ksteps; ++t) {
                const uint32_t k = b * ksteps + t;
                matmul_tiles(cb_x, cb_w, k, 2 * t, 0);
                matmul_tiles(cb_x, cb_w, k, 2 * t + 1, 1);
            }
            cb_pop_front(cb_w, 2 * ksteps);
        }
        reconfig_data_format_srca(cb_w, cb_sc);
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
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);
    }
    cb_pop_front(cb_sc, 2 * slots);
    cb_pop_front(cb_x, kx);
}
