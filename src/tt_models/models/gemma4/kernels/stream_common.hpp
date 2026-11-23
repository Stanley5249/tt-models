// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Adapted from tenstorrent/tt-metal@d1a102793682817a4c12c572c44e6ad0277b5fe0
// models/demos/gpt_oss/tt/fused_decode/kernels/stream_*_reader.cpp.
// Block streaming from one DRAM bank with transaction ids (NCRISC), after the gpt-oss fused
// decode streamers.
//
// A block is `packets` NOC packets of `packet_bytes`, pushed to `cb` as `block_tiles` tiles once all its
// packets landed. Three block buffers, one block in flight ahead of the one compute consumes.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <uint32_t cb, uint32_t block_tiles, uint32_t block_bytes>
struct BlockStream {
    static constexpr uint32_t num_buffers = 3;
    static constexpr uint32_t extra_in_flight = 1;
    uint64_t base;
    uint32_t l1_write, cb_base, cb_end, free_blocks, trid, trid_to_wait;

    FORCE_INLINE void init(uint64_t bank_base, uint32_t packet_bytes, uint32_t vc) {
        base = bank_base;
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        noc_async_read_one_packet_set_state<true>(base, packet_bytes, vc);
        cb_reserve_back(cb, block_tiles * (extra_in_flight + 1));
        l1_write = get_write_ptr(cb);
        cb_base = l1_write;
        cb_end = cb_base + num_buffers * block_bytes;
        free_blocks = num_buffers;
        trid = 1;
        trid_to_wait = 1;
    }

    FORCE_INLINE void begin_block() { noc_async_read_set_trid(trid); }

    FORCE_INLINE void packet(uint32_t src_offset, uint32_t packet_bytes) {
        noc_async_read_one_packet_with_state_with_trid(base, src_offset, l1_write, trid);
        l1_write += packet_bytes;
    }

    FORCE_INLINE void end_block() {
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb, block_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb, block_tiles * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }

    FORCE_INLINE void finish() {
        for (uint32_t i = 0; i < extra_in_flight; ++i) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb, block_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
        }
    }
};

// Reads the first 32 bytes of row-major page 0 of a tensor (the routed expert ids or weights of token 0).
template <typename Accessor>
FORCE_INLINE void read_row0(const Accessor& s, uint32_t l1, uint32_t bytes) {
    noc_async_read(s.get_noc_addr(0), l1, bytes);
}

// Row 0 of a 32x32 BF16 tile is 16 values at byte 0 (face 0) and 16 values at byte 512 (face 1).
constexpr uint32_t ROW0_FACE_BYTES = 32;
constexpr uint32_t FACE1_OFFSET = 512;
constexpr uint32_t TILE_BYTES_BF16 = 2048;

// The contiguous run of the n routed slots that core `rank` of `readers` takes: ceil(n / readers) slots
// each, so the runs stay balanced whatever n is.
struct SlotRun {
    uint32_t lo, hi;
};

FORCE_INLINE SlotRun slot_run(uint32_t n, uint32_t readers, uint32_t rank) {
    const uint32_t per = (n + readers - 1) / readers;
    const uint32_t lo = rank * per < n ? rank * per : n;
    return {lo, lo + per < n ? lo + per : n};
}
