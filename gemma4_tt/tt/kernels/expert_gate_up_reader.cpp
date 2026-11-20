// Routed-expert gate|up weight streamer (NCRISC, NOC0) for Gemma 4 batch-1 decode.
//
// The per-chip expert gate|up weight is ttnn's interleaved [128, 2816, 768] BFP4 tensor (tile columns
// 0..10 gate, 11 zero, 12..22 up, 23 zero). Interleaving puts tile page p on DRAM bank p % 12, and 24
// columns are a multiple of 12, so bank b holds column pair (gate b, up b) of every expert, stored
// k-major: [k0: gate, up][k1: gate, up]... one contiguous 176-tile block per expert at bank page e * 176.
// This core streams that block for its routed slots in blocks of 22 tiles (11 k-steps).
//
// split = 1 (the dense MLP gate|up, [2816, 2304] BFP8 per chip): bank b holds 6 columns per K row,
// gate b + 12 j at row offset j and up 36 + b + 12 j at offset j + up_delta; this core reads its pair j
// as two one-tile packets per k-step, k_stride bytes apart.
//
// runtime args: [w_addr, idx_addr, bank_id, vc, slot0, pair_j]

#include "stream_common.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t slot0 = get_arg_val<uint32_t>(4);
    const uint32_t pair_j = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);       // blocks per expert
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);  // tiles per block
    constexpr uint32_t packets = get_compile_time_arg_val(5);      // packets per block
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t expert_bytes = get_compile_time_arg_val(7);  // one expert's block in this bank
    constexpr uint32_t split = get_compile_time_arg_val(8);
    constexpr uint32_t k_stride = get_compile_time_arg_val(9);  // split: bytes per K row of the bank
    constexpr uint32_t up_delta = get_compile_time_arg_val(10);  // split: up tile offset from gate, in tiles
    constexpr uint32_t id_bytes = get_compile_time_arg_val(11);  // the ids row, <= 64
    constexpr auto idx_args = TensorAccessorArgs<12>();

    constexpr uint32_t block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t packet_bytes = split ? tile_bytes : block_bytes / packets;
    static_assert(split || packet_bytes * packets == block_bytes, "packets must split the block");
    static_assert(packet_bytes <= NOC_MAX_BURST_SIZE, "packet over the NOC burst size");

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, id_bytes);
    read_row0(s_idx, scr, id_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    uint32_t offsets[slots];
    for (uint32_t i = 0; i < slots; ++i) {
        offsets[i] = static_cast<uint32_t>(ids[slot0 + i]) * expert_bytes;
    }

    BlockStream<cb_w, block_tiles, block_bytes> st;
    st.init(get_noc_addr_from_bank_id<true>(bank_id, w_addr), packet_bytes, vc);
    for (uint32_t i = 0; i < slots; ++i) {
        uint32_t src = offsets[i];
        if constexpr (split) {
            src += pair_j * tile_bytes;
            for (uint32_t b = 0; b < blocks; ++b) {
                st.begin_block();
                for (uint32_t k = 0; k < block_tiles / 2; ++k) {
                    st.packet(src, tile_bytes);
                    st.packet(src + up_delta * tile_bytes, tile_bytes);
                    src += k_stride;
                }
                st.end_block();
            }
        } else {
            for (uint32_t b = 0; b < blocks; ++b) {
                st.begin_block();
                for (uint32_t p = 0; p < packets; ++p) {
                    st.packet(src, packet_bytes);
                    src += packet_bytes;
                }
                st.end_block();
            }
        }
    }
    st.finish();
}
