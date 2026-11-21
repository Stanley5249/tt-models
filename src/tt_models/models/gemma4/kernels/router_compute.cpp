// Router math (TRISC). Every core: partial logits over its K slice, P[n] = sum_k x[k] . W[k, n] (FP32).
// The reducer then sums the partial rows gathered in cb_part (J @ P puts each column's total in every
// row) and takes exp, which the writer uses for the top-k and its softmax.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_j = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = get_compile_time_arg_val(4);
    constexpr uint32_t cb_e = get_compile_time_arg_val(5);
    constexpr uint32_t kt = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);
    const uint32_t is_reducer = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_partial);
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_w, kt * n_tiles);
    tile_regs_acquire();
    for (uint32_t k = 0; k < kt; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            matmul_tiles(cb_x, cb_w, k, k * n_tiles + n, n);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_partial, n_tiles);
    for (uint32_t n = 0; n < n_tiles; ++n) {
        pack_tile(n, cb_partial);
    }
    cb_push_back(cb_partial, n_tiles);
    tile_regs_release();
    cb_pop_front(cb_x, kt);
    cb_pop_front(cb_w, kt * n_tiles);

    if (is_reducer) {
        cb_wait_front(cb_j, 1);
        cb_wait_front(cb_part, n_tiles);
        reconfig_data_format(cb_part, cb_j);  // FP32 operands from here on
        matmul_init(cb_j, cb_part);
        tile_regs_acquire();
        for (uint32_t n = 0; n < n_tiles; ++n) {
            matmul_tiles(cb_j, cb_part, 0, n, n);
        }
        exp_tile_init();
        for (uint32_t n = 0; n < n_tiles; ++n) {
            exp_tile(n);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_e, n_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            pack_tile(n, cb_e);
        }
        cb_push_back(cb_e, n_tiles);
        tile_regs_release();
        cb_pop_front(cb_part, n_tiles);
        cb_pop_front(cb_j, 1);
    }
}
