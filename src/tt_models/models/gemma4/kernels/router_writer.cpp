// Router reduction and top-k (BRISC).
//
// Every core writes rows 0 .. rows - 1 (one per token) of its tn FP32 partial logit tiles (expert tiles
// n0 ..) into rows slot * rows + b of the reducer's cb_part tiles and increments the reducer's semaphore. The reducer zeroes
// the unused rows, waits for all partials, hands cb_part to compute (J @ partials puts token b's logits in
// row b), and from row b of exp(logits) picks token b's top k (positive floats compare as integers),
// normalizes them in software float (softmax over the k logits, as Gemma's softmax-top-k-renormalize).
//
// Outputs, row-major: ids (uint16) holds the step's distinct experts, one slot each in order of first
// pick, and their count at entry `slots` (the most slots the rows can need); weights (BF16) is
// slot-major [slot * rows + b], token b's weight for that slot's expert or zero. A batch-1 step thus
// gets its k experts in top-k order, as before.
//
// runtime args: [slot, reducer_x, reducer_y, is_reducer, ids_addr, wts_addr, n0]

#include "compact.hpp"

void kernel_main() {
    const uint32_t slot = get_arg_val<uint32_t>(0);
    const uint32_t red_x = get_arg_val<uint32_t>(1);
    const uint32_t red_y = get_arg_val<uint32_t>(2);
    const uint32_t is_reducer = get_arg_val<uint32_t>(3);
    const uint32_t ids_addr = get_arg_val<uint32_t>(4);
    const uint32_t wts_addr = get_arg_val<uint32_t>(5);
    const uint32_t n0 = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_e = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t n_parts = get_compile_time_arg_val(5);
    constexpr uint32_t top_k = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t rows = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);      // min(experts, rows * k)
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(10);  // bytes of the ids row written
    constexpr uint32_t wts_bytes = get_compile_time_arg_val(11);  // bytes of the weights row written
    constexpr uint32_t tn = get_compile_time_arg_val(12);       // expert tiles per core
    constexpr uint32_t n_cores = get_compile_time_arg_val(13);  // n_parts * n_tiles / tn
    constexpr auto ids_args = TensorAccessorArgs<14>();
    static_assert(n_parts * rows <= 32 && rows <= 16, "partial rows fit a tile, token rows its top faces");
    static_assert(2 * (slots + 1) <= ids_bytes && 2 * slots * rows <= wts_bytes, "outputs fit their rows");
    constexpr auto wts_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t face = 1024;  // FP32

    const uint32_t sem_addr = get_semaphore(sem_id);
    cb_reserve_back(cb_part, n_tiles);
    const uint32_t part = get_write_ptr(cb_part);  // same address on every core
    if (is_reducer) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t r = n_parts * rows; r < 32; ++r) {
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

    cb_wait_front(cb_partial, tn);
    const uint32_t src = get_read_ptr(cb_partial);
    for (uint32_t b = 0; b < rows; ++b) {
        const uint32_t r = slot * rows + b;
        const uint32_t row_off = r < 16 ? r * 64 : 2 * face + (r - 16) * 64;
        for (uint32_t n = 0; n < tn; ++n) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint64_t dst =
                    get_noc_addr(red_x, red_y, part + (n0 + n) * COMPACT_TILE + f * face + row_off);
                noc_async_write(src + n * COMPACT_TILE + f * face + b * 64, dst, 64);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_partial, tn);
    noc_semaphore_inc(get_noc_addr(red_x, red_y, sem_addr), 1);
    noc_async_atomic_barrier();
    if (!is_reducer) {
        return;
    }

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, n_cores);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_part, n_tiles);

    cb_wait_front(cb_e, n_tiles);
    const uint32_t e = get_read_ptr(cb_e);
    cb_reserve_back(cb_out, 1);
    const uint32_t out = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out + ids_bytes);
    for (uint32_t j = 0; j < ids_bytes / 2; ++j) {
        ids[j] = 0;
    }
    for (uint32_t j = 0; j < wts_bytes / 2; ++j) {
        wts[j] = 0;
    }
    constexpr uint32_t no_slot = 0xFF;
    uint8_t slot_of[n_tiles * 32];
    for (uint32_t e = 0; e < n_tiles * 32; ++e) {
        slot_of[e] = no_slot;
    }
    uint32_t used = 0;
    for (uint32_t b = 0; b < rows; ++b) {
        uint32_t best_v[top_k], best_i[top_k];
        for (uint32_t j = 0; j < top_k; ++j) {
            best_v[j] = 0;
            best_i[j] = 0;
        }
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* row =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e + n * COMPACT_TILE + f * face + b * 64);
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
        const float inv = 1.0f / sum;
        for (uint32_t j = 0; j < top_k; ++j) {
            union {
                uint32_t u;
                float f;
            } w;
            w.f = vals[j] * inv;
            uint32_t s = slot_of[best_i[j]];
            if (s == no_slot) {
                s = used++;
                slot_of[best_i[j]] = s;
                ids[s] = static_cast<uint16_t>(best_i[j]);
            }
            wts[s * rows + b] = static_cast<uint16_t>((w.u + 0x7FFF + ((w.u >> 16) & 1)) >> 16);
        }
    }
    ids[slots] = static_cast<uint16_t>(used);
    cb_pop_front(cb_e, n_tiles);
    const auto s_ids = TensorAccessor(ids_args, ids_addr, ids_bytes);
    const auto s_wts = TensorAccessor(wts_args, wts_addr, wts_bytes);
    noc_async_write(out, s_ids.get_noc_addr(0), ids_bytes);
    noc_async_write(out + ids_bytes, s_wts.get_noc_addr(0), wts_bytes);
    noc_async_write_barrier();
}
