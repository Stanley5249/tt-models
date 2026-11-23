// LM head weight streamer (NCRISC, NOC0).
//
// The per-chip LM head weight is an interleaved [2816, 12 * bank_cols * 32] BFP8 tensor whose row of
// tiles is a multiple of the 12 DRAM banks, so bank b holds tile columns b + 12 j (j < bank_cols) of every
// K row, contiguous per row. This core owns columns j0 .. j0 + groups * group - 1 of its bank and streams
// them group by group: per group, kt / block_k blocks of block_k K rows of `group` tiles (one packet per
// K row).
//
// runtime args: [w_addr, bank_id, vc, j0, groups]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t j0 = get_arg_val<uint32_t>(3);
    const uint32_t groups = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t bank_cols = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_k = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);
    static_assert(kt % block_k == 0, "blocks must split K");

    constexpr uint32_t packet_bytes = group * tile_bytes;
    constexpr uint32_t block_tiles = block_k * group;
    constexpr uint32_t row_bytes = bank_cols * tile_bytes;

    BlockStream<cb_w, block_tiles, block_tiles * tile_bytes> st;
    st.init(get_noc_addr_from_bank_id<true>(bank_id, w_addr), packet_bytes, vc);
    for (uint32_t g = 0; g < groups; ++g) {
        uint32_t src = (j0 + g * group) * tile_bytes;
        for (uint32_t b = 0; b < kt / block_k; ++b) {
            st.begin_block();
            for (uint32_t k = 0; k < block_k; ++k) {
                st.packet(src, packet_bytes);
                src += row_bytes;
            }
            st.end_block();
        }
    }
    st.finish();
}
