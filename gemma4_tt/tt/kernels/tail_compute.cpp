// Layer-boundary math (TRISC) on compact FP32 vectors of V tiles (gemma4_tt/tt/boundary.py).
//
// RMS statistics use two matmuls against an all-ones tile J: (A*A) @ J puts every row sum in its row,
// J @ that puts the total in every element, so 1 / sqrt(mean + eps) multiplies elementwise without a
// broadcast. Padding chunks are zero in every input and weight, so they add nothing.
//
// ATTN_TAIL: a = g0 + g1; x2 = x + rms(a) * w_post_attn; n = rms(x2);
//            outputs x2 (FP32), n, n * w_pre_ff, n * w_pre_ff2 (BF16)
// FF_TAIL:   m = g0m + g1m; e = g0e + g1e; t = rms(m) * w1 + rms(e) * w2;
//            x3 = (x2 + rms(t) * w_ff) * layer_scalar; outputs x3 (FP32), rms(x3) * w_in (BF16)

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_g = get_compile_time_arg_val(0);
constexpr uint32_t cb_x = get_compile_time_arg_val(1);
constexpr uint32_t cb_w = get_compile_time_arg_val(2);
constexpr uint32_t cb_j = get_compile_time_arg_val(3);
constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
constexpr uint32_t cb_rs = get_compile_time_arg_val(5);
constexpr uint32_t cb_r = get_compile_time_arg_val(6);
constexpr uint32_t cb_t0 = get_compile_time_arg_val(7);  // FP32 scratch vectors cb_t0 .. cb_t0 + 3
constexpr uint32_t cb_xo = get_compile_time_arg_val(8);  // FP32 compact residual output
constexpr uint32_t cb_row0 = get_compile_time_arg_val(9);  // BF16 outputs cb_row0, cb_row0 + 1, ...
constexpr uint32_t V = get_compile_time_arg_val(10);
constexpr uint32_t inv_h = get_compile_time_arg_val(11);  // FP32 bits
constexpr uint32_t eps = get_compile_time_arg_val(12);
constexpr uint32_t scalar = get_compile_time_arg_val(13);  // layer scalar (FF_TAIL)
constexpr uint32_t g_half = get_compile_time_arg_val(14);  // tiles per chip in cb_g
constexpr uint32_t w_tiles = get_compile_time_arg_val(15);

// Pack dest tiles 0..V-1 to `cb` (FP32 or BF16; the packer format follows `cb`).
ALWI void pack_vec(uint32_t cb) {
    cb_reserve_back(cb, V);
    for (uint32_t t = 0; t < V; ++t) {
        pack_tile(t, cb);
    }
    cb_push_back(cb, V);
}

// Packs the acquired dest tiles to one or two CBs, then releases dest.
ALWI void emit(uint32_t cb_a, uint32_t cb_b = 0xFFFF) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_a);
    pack_vec(cb_a);
    if (cb_b != 0xFFFF) {
        pack_reconfig_data_format(cb_b);
        pack_vec(cb_b);
    }
    tile_regs_release();
}

// dest[t] = A[a + t] + B[b + t]
ALWI void add_vec(uint32_t cba, uint32_t a, uint32_t cbb, uint32_t b) {
    add_init(cba, cbb, false);
    tile_regs_acquire();
    for (uint32_t t = 0; t < V; ++t) {
        add_tiles(cba, cbb, a + t, b + t, t);
    }
}

// dest[t] = A[a + t] * B[b + t]
ALWI void mul_vec(uint32_t cba, uint32_t a, uint32_t cbb, uint32_t b) {
    mul_init(cba, cbb, false);
    tile_regs_acquire();
    for (uint32_t t = 0; t < V; ++t) {
        mul_tiles(cba, cbb, a + t, b + t, t);
    }
}

// cb_r <- one tile holding 1 / sqrt(mean(A^2) + eps) in every element.
ALWI void rms_scale(uint32_t cba, uint32_t a) {
    mul_vec(cba, a, cba, a);
    emit(cb_sq);
    cb_wait_front(cb_sq, V);
    matmul_init(cb_sq, cb_j);
    tile_regs_acquire();
    for (uint32_t t = 0; t < V; ++t) {
        matmul_tiles(cb_sq, cb_j, t, 0, 0);  // row sums, accumulated over the V tiles
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_rs);
    cb_reserve_back(cb_rs, 1);
    pack_tile(0, cb_rs);
    cb_push_back(cb_rs, 1);
    tile_regs_release();
    cb_pop_front(cb_sq, V);

    cb_wait_front(cb_rs, 1);
    matmul_init(cb_j, cb_rs);
    tile_regs_acquire();
    matmul_tiles(cb_j, cb_rs, 0, 0, 0);  // total in every element
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_r);
    cb_reserve_back(cb_r, 1);
    pack_tile(0, cb_r);
    cb_push_back(cb_r, 1);
    tile_regs_release();
    cb_pop_front(cb_rs, 1);
}

// dest[t] = A[a + t] * r (* W[w + t] when w_cb is a CB). Pops cb_r.
ALWI void norm_vec(uint32_t cba, uint32_t a, uint32_t w_cb, uint32_t w) {
    rms_scale(cba, a);
    cb_wait_front(cb_r, 1);
    mul_init(cba, cb_r, false);
    tile_regs_acquire();
    for (uint32_t t = 0; t < V; ++t) {
        mul_tiles(cba, cb_r, a + t, 0, t);
    }
    if (w_cb != 0xFFFF) {
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w_cb);
        for (uint32_t t = 0; t < V; ++t) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w_cb, w + t, t);
        }
    }
    cb_pop_front(cb_r, 1);
}

void kernel_main() {
    compute_kernel_hw_startup(cb_g, cb_x, cb_xo);
    constexpr uint32_t t_a = cb_t0, t_b = cb_t0 + 1, t_c = cb_t0 + 2, t_d = cb_t0 + 3;
    cb_wait_front(cb_j, 1);
    cb_wait_front(cb_g, 2 * g_half);
    cb_wait_front(cb_x, V);
#if defined(ATTN_TAIL)
    // weights: [post_attn | pre_ff | pre_ff2], V tiles each
    add_vec(cb_g, 0, cb_g, g_half);
    emit(t_a);  // a
    cb_wait_front(t_a, V);
    cb_wait_front(cb_w, w_tiles);
    norm_vec(t_a, 0, cb_w, 0);
    emit(t_b);  // rms(a) * w_post_attn
    cb_pop_front(t_a, V);
    cb_wait_front(t_b, V);
    add_vec(cb_x, 0, t_b, 0);
    emit(cb_xo, t_c);  // x2
    cb_pop_front(t_b, V);
    cb_wait_front(t_c, V);
    norm_vec(t_c, 0, 0xFFFF, 0);
    emit(cb_row0, t_d);  // n = rms(x2)
    cb_pop_front(t_c, V);
    cb_wait_front(t_d, V);
    mul_vec(t_d, 0, cb_w, V);
    emit(cb_row0 + 1);  // n * w_pre_ff
    mul_vec(t_d, 0, cb_w, 2 * V);
    emit(cb_row0 + 2);  // n * w_pre_ff2
    cb_pop_front(t_d, V);
#elif defined(FF_TAIL)
    // cb_g per chip: [mlp (V) | experts (V)]; weights: [post_ff1 | post_ff2 | post_ff | next input_ln]
    add_vec(cb_g, 0, cb_g, g_half);
    emit(t_a);  // m
    add_vec(cb_g, V, cb_g, g_half + V);
    emit(t_b);  // e
    cb_wait_front(t_a, V);
    cb_wait_front(cb_w, w_tiles);
    norm_vec(t_a, 0, cb_w, 0);
    emit(t_c);  // rms(m) * w1
    cb_pop_front(t_a, V);
    cb_wait_front(t_b, V);
    norm_vec(t_b, 0, cb_w, V);
    emit(t_d);  // rms(e) * w2
    cb_pop_front(t_b, V);
    cb_wait_front(t_c, V);
    cb_wait_front(t_d, V);
    add_vec(t_c, 0, t_d, 0);
    emit(t_a);  // t
    cb_pop_front(t_c, V);
    cb_pop_front(t_d, V);
    cb_wait_front(t_a, V);
    norm_vec(t_a, 0, cb_w, 2 * V);
    emit(t_b);  // rms(t) * w_ff
    cb_pop_front(t_a, V);
    cb_wait_front(t_b, V);
    add_vec(cb_x, 0, t_b, 0);
    binop_with_scalar_tile_init();
    for (uint32_t t = 0; t < V; ++t) {
        mul_unary_tile(t, scalar);
    }
    emit(cb_xo, t_c);  // x3
    cb_pop_front(t_b, V);
    cb_wait_front(t_c, V);
    norm_vec(t_c, 0, cb_w, 3 * V);
    emit(cb_row0);  // rms(x3) * w_in
    cb_pop_front(t_c, V);
#endif
    cb_pop_front(cb_g, 2 * g_half);
    cb_pop_front(cb_x, V);
    cb_pop_front(cb_w, w_tiles);
    cb_pop_front(cb_j, 1);
}
