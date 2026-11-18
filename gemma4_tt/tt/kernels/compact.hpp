// Compact layout of one decode hidden vector (gemma4_tt/tt/boundary.py).
//
// Row form: the vector is row 0 of [32, W] tiles. Its 16-value chunk c sits in tile c / 2, face c % 2,
// row 0 (byte (c % 2) * face_bytes). Compact form: FP32 chunk c (64 bytes) at byte 64 c of a tile sequence,
// 64 chunks per 32x32 FP32 tile, so the vector is a few dense tiles. Padding chunks are zero.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t COMPACT_CHUNK = 64;     // 16 FP32 values
constexpr uint32_t COMPACT_TILE = 4096;    // FP32 tile
constexpr uint32_t CHUNKS_PER_TILE = 64;

template <uint32_t elem_bytes>
struct RowForm {
    static constexpr uint32_t chunk_bytes = 16 * elem_bytes;
    static constexpr uint32_t face_bytes = 256 * elem_bytes;
    static constexpr uint32_t tile_bytes = 1024 * elem_bytes;
};

template <typename Accessor>
FORCE_INLINE uint64_t compact_chunk_addr(const Accessor& s, uint32_t chunk) {
    return s.get_noc_addr(chunk / CHUNKS_PER_TILE, (chunk % CHUNKS_PER_TILE) * COMPACT_CHUNK);
}

template <uint32_t elem_bytes, typename Accessor>
FORCE_INLINE uint64_t row_chunk_addr(const Accessor& s, uint32_t chunk) {
    return s.get_noc_addr(chunk / 2, (chunk % 2) * RowForm<elem_bytes>::face_bytes);
}
