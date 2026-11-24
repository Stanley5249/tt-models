// Find the first maximum in this core's vocabulary slice of each FP32 row.
#include "api/dataflow/dataflow_api.h"
#include "api/numeric/float32.h"

// Compare the FP32 scores as unsigned integers, with both zeros equal and NaNs first.
FORCE_INLINE uint32_t score_key(uint32_t bits) {
    const uint32_t magnitude = bits & FLOAT32_MAGNITUDE_MASK;
    if (magnitude > POS_INF_FLOAT32) {
        return 0xFFFFFFFF;
    }
    if (magnitude == 0) {
        return FLOAT32_SIGN_MASK;
    }
    return bits ^ ((bits & FLOAT32_SIGN_MASK) ? 0xFFFFFFFF : FLOAT32_SIGN_MASK);
}

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t candidates_addr = get_arg_val<uint32_t>(1);
    const uint32_t worker = get_arg_val<uint32_t>(2);
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t slice = get_compile_time_arg_val(1);
    constexpr uint32_t vocab = get_compile_time_arg_val(2);
    constexpr uint32_t result_bytes = get_compile_time_arg_val(3);
    constexpr auto logits_args = TensorAccessorArgs<4>();
    constexpr auto candidates_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(logits_args, logits_addr, vocab * 4);
    const auto candidates = TensorAccessor(candidates_args, candidates_addr, result_bytes);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t result = get_write_ptr(1);
    const auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto* pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result);
    const uint32_t start = worker * slice;

    for (uint32_t row = 0; row < rows; ++row) {
        noc_async_read(logits.get_noc_addr(row, start * 4), scratch, slice * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        uint32_t best_key = 0;
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < slice; ++i) {
            const uint32_t key = score_key(values[i]);
            if (key > best_key) {
                best_key = key;
                best_id = i;
            }
        }
        pairs[2 * row] = best_key;
        pairs[2 * row + 1] = start + best_id;
    }
    for (uint32_t i = 2 * rows; i < result_bytes / 4; ++i) {
        pairs[i] = 0;
    }
    noc_async_write_page(worker, candidates, result);
    noc_async_write_barrier();
}
