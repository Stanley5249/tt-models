// Routed-expert down weight streamer (NCRISC, NOC0) for Gemma 4 batch-1 decode.
//
// The per-chip down weight is an interleaved [128, 352, 3072] BFP4 tensor (hidden padded 2816 -> 3072 so
// that the 96 column tiles are a multiple of the 12 DRAM banks): bank b holds columns n = b + 12 j
// (j = 0..7) of every expert, k-major, at bank page (e * 11 + k) * 8 + j. This core owns columns
// j0 .. j0 + J - 1 of its bank; per routed slot it reads one J-tile packet per k (one block per slot).
//
// runtime args: [w_addr, idx_addr, bank_id, vc, j0]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t j0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);       // J
    constexpr uint32_t bank_cols = get_compile_time_arg_val(5);  // columns per bank (8)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr auto idx_args = TensorAccessorArgs<7>();

    constexpr uint32_t packet_bytes = cols * tile_bytes;
    constexpr uint32_t k_stride = bank_cols * tile_bytes;
    constexpr uint32_t expert_bytes = kt * k_stride;
    constexpr uint32_t block_tiles = kt * cols;

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, 32);
    read_row0(s_idx, scr, 32);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);

    BlockStream<cb_w, block_tiles, block_tiles * tile_bytes> st;
    st.init(get_noc_addr_from_bank_id<true>(bank_id, w_addr), packet_bytes, vc);
    for (uint32_t e = 0; e < slots; ++e) {
        uint32_t src = static_cast<uint32_t>(ids[e]) * expert_bytes + j0 * tile_bytes;
        st.begin_block();
        for (uint32_t k = 0; k < kt; ++k) {
            st.packet(src, packet_bytes);
            src += k_stride;
        }
        st.end_block();
    }
    st.finish();
}
