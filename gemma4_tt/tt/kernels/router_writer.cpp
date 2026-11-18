// Router reduction and top-k (BRISC).
//
// Every core writes row 0 of its FP32 partial logit tiles into row `slot` of the reducer's cb_part tiles
// and increments the reducer's semaphore. The reducer zeroes the unused rows, waits for all partials,
// hands cb_part to compute, and from exp(logits) picks the top k (positive floats compare as integers),
// normalizes them in software float (softmax over the k logits, as Gemma's softmax-top-k-renormalize),
// and writes ids (uint16) and weights (BF16) as row-major rows.
//
// runtime args: [slot, reducer_x, reducer_y, is_reducer, ids_addr, wts_addr]

#include "compact.hpp"

void kernel_main() {
    const uint32_t slot = get_arg_val<uint32_t>(0);
    const uint32_t red_x = get_arg_val<uint32_t>(1);
    const uint32_t red_y = get_arg_val<uint32_t>(2);
    const uint32_t is_reducer = get_arg_val<uint32_t>(3);
    const uint32_t ids_addr = get_arg_val<uint32_t>(4);
    const uint32_t wts_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_e = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t n_parts = get_compile_time_arg_val(5);
    constexpr uint32_t top_k = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr auto ids_args = TensorAccessorArgs<8>();
    constexpr auto wts_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t face = 1024;  // FP32

    const uint32_t sem_addr = get_semaphore(sem_id);
    cb_reserve_back(cb_part, n_tiles);
    const uint32_t part = get_write_ptr(cb_part);  // same address on every core
    if (is_reducer) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t r = n_parts; r < 32; ++r) {
                const uint32_t off = n * COMPACT_TILE + (r < 16 ? r * 64 : 2 * face + (r - 16) * 64);  // faces 0 / 2
                volatile tt_l1_ptr uint32_t* p0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part + off);
                volatile tt_l1_ptr uint32_t* p1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part + off + face);
                for (uint32_t i = 0; i < 16; ++i) {
                    p0[i] = 0;
                    p1[i] = 0;
                }
            }
        }
    }

    cb_wait_front(cb_partial, n_tiles);
    const uint32_t src = get_read_ptr(cb_partial);
    const uint32_t row_off = slot < 16 ? slot * 64 : 2 * face + (slot - 16) * 64;
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t f = 0; f < 2; ++f) {
            const uint64_t dst = get_noc_addr(red_x, red_y, part + n * COMPACT_TILE + f * face + row_off);
            noc_async_write(src + n * COMPACT_TILE + f * face, dst, 64);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_partial, n_tiles);
    noc_semaphore_inc(get_noc_addr(red_x, red_y, sem_addr), 1);
    noc_async_atomic_barrier();
    if (!is_reducer) {
        return;
    }

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, n_parts);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_part, n_tiles);

    cb_wait_front(cb_e, n_tiles);
    const uint32_t e = get_read_ptr(cb_e);
    uint32_t best_v[top_k], best_i[top_k];
    for (uint32_t j = 0; j < top_k; ++j) {
        best_v[j] = 0;
        best_i[j] = 0;
    }
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e + n * COMPACT_TILE + f * face);
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t v = row[c];
                if (v <= best_v[top_k - 1]) {
                    continue;
                }
                uint32_t j = top_k - 1;
                while (j > 0 && best_v[j - 1] < v) {
                    best_v[j] = best_v[j - 1];
                    best_i[j] = best_i[j - 1];
                    --j;
                }
                best_v[j] = v;
                best_i[j] = n * 32 + f * 16 + c;
            }
        }
    }
    cb_pop_front(cb_e, n_tiles);

    float sum = 0.0f;
    float vals[top_k];
    for (uint32_t j = 0; j < top_k; ++j) {
        union {
            uint32_t u;
            float f;
        } cv;
        cv.u = best_v[j];
        vals[j] = cv.f;
        sum += cv.f;
    }
    cb_reserve_back(cb_out, 1);
    const uint32_t out = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out + 64);
    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < 16; ++j) {
        if (j < top_k) {
            union {
                uint32_t u;
                float f;
            } w;
            w.f = vals[j] * inv;
            ids[j] = static_cast<uint16_t>(best_i[j]);
            wts[j] = static_cast<uint16_t>((w.u + 0x7FFF + ((w.u >> 16) & 1)) >> 16);
        } else {
            ids[j] = 0;
            wts[j] = 0;
        }
    }
    const auto s_ids = TensorAccessor(ids_args, ids_addr, 32);
    const auto s_wts = TensorAccessor(wts_args, wts_addr, 32);
    noc_async_write(out, s_ids.get_noc_addr(0), 32);
    noc_async_write(out + 64, s_wts.get_noc_addr(0), 32);
    noc_async_write_barrier();
}
