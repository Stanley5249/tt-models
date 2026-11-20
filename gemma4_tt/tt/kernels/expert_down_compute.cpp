// Routed-expert down matmul (TRISC) for Gemma 4 decode: for each of this core's J output columns,
// out = sum over slots e and K tiles k of act[e, k] . Wd_e[k, col], accumulated in FP32 dest. A slot's
// act tiles arrive as kt tiles of cb_in0 (the writer streams them slot by slot).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t slots = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);
    matmul_init(cb_in0, cb_w);
    tile_regs_acquire();
    for (uint32_t e = 0; e < slots; ++e) {
        cb_wait_front(cb_in0, kt);
        cb_wait_front(cb_w, kt * cols);
        for (uint32_t k = 0; k < kt; ++k) {
            for (uint32_t c = 0; c < cols; ++c) {
                matmul_tiles(cb_in0, cb_w, k, k * cols + c, c);
            }
        }
        cb_pop_front(cb_w, kt * cols);
        cb_pop_front(cb_in0, kt);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t c = 0; c < cols; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
}
