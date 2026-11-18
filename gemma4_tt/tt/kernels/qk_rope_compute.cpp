// q/k norm + RoPE (TRISC) on one compact tile of the fused QKV row (boundary.py QkRope):
//   out = rsqrt(mean_head(x^2) + eps) * (x * w * cos + xr * wr * sin_signed)
// The per-head sums of x^2 are B @ ((x * x) @ A) with block-ones A and B (a head is one or two faces).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_s1 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_r = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);  // BF16
    constexpr uint32_t inv_hd = get_compile_time_arg_val(10);
    constexpr uint32_t eps = get_compile_time_arg_val(11);

    compute_kernel_hw_startup(cb_x, cb_w, cb_sq);
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_xr, 1);
    cb_wait_front(cb_cs, 2);
    cb_wait_front(cb_w, 2);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);

    // sq = x * x
    mul_init(cb_x, cb_x, false);
    tile_regs_acquire();
    mul_tiles(cb_x, cb_x, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sq, 1);
    pack_tile(0, cb_sq);
    cb_push_back(cb_sq, 1);
    tile_regs_release();

    // s1 = sq @ A
    cb_wait_front(cb_sq, 1);
    matmul_init(cb_sq, cb_a);
    tile_regs_acquire();
    matmul_tiles(cb_sq, cb_a, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_s1, 1);
    pack_tile(0, cb_s1);
    cb_push_back(cb_s1, 1);
    tile_regs_release();
    cb_pop_front(cb_sq, 1);

    // r = rsqrt(B @ s1 / hd + eps)
    cb_wait_front(cb_s1, 1);
    matmul_init(cb_b, cb_s1);
    tile_regs_acquire();
    matmul_tiles(cb_b, cb_s1, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_hd);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_r, 1);
    pack_tile(0, cb_r);
    cb_push_back(cb_r, 1);
    tile_regs_release();
    cb_pop_front(cb_s1, 1);

    // dst0 = x * w * cos; dst1 = xr * wr * sin; out = (dst0 + dst1) * r
    cb_wait_front(cb_r, 1);
    mul_init(cb_x, cb_w, false);
    tile_regs_acquire();
    mul_tiles(cb_x, cb_w, 0, 0, 0);
    mul_tiles(cb_xr, cb_w, 0, 1, 1);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, 0, 0);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sq, 2);
    pack_tile(0, cb_sq);
    pack_tile(1, cb_sq);
    cb_push_back(cb_sq, 2);
    tile_regs_release();

    cb_wait_front(cb_sq, 2);
    add_init(cb_sq, cb_sq, false);
    tile_regs_acquire();
    add_tiles(cb_sq, cb_sq, 0, 1, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_r);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_r, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_sq, 2);
    cb_pop_front(cb_r, 1);

    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_xr, 1);
    cb_pop_front(cb_cs, 2);
    cb_pop_front(cb_w, 2);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
