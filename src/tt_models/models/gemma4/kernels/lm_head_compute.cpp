// LM head matmul (TRISC): for each group of `group` output column tiles, out = sum over the kt K tiles of
// x[k] . W[k, col], accumulated in FP32 dest and packed as FP32 tiles.
//
// runtime args: [groups]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t block_k = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);
    const uint32_t groups = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);
    matmul_init(cb_in0, cb_w);
    cb_wait_front(cb_in0, kt);
    for (uint32_t g = 0; g < groups; ++g) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / block_k; ++b) {
            cb_wait_front(cb_w, block_k * group);
            for (uint32_t k = 0; k < block_k; ++k) {
                for (uint32_t c = 0; c < group; ++c) {
                    matmul_tiles(cb_in0, cb_w, b * block_k + k, k * group + c, c);
                }
            }
            cb_pop_front(cb_w, block_k * group);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, group);
        tile_regs_wait();
        for (uint32_t c = 0; c < group; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, group);
    }
    cb_pop_front(cb_in0, kt);
}
