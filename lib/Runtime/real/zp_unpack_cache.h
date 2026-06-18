/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H
#define HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H

// Pointer-keyed cache of unpacked MatMulNBits zero_points buffers. Used by both
// wrap_matmul_nbits and wrap_qmoe to avoid re-launching the unpack/convert
// kernels on every call when zero_points pointers are stable across inferences
// (which they are — the pointers come from the model constants blob).
//
// Ownership (see docs/design/op-state-slots-design.md): matmul_nbits owns a
// per-op-instance cache in its MatmulNbitsState op-state slot; qmoe uses the
// per-session RuntimeState::zp_unpack_cache (one per session, shared across all
// qmoe instances and reached via get_or_create_zp_cache). Each zero_points
// pointer is distinct, so entries never collide across ops.

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>

struct RuntimeState;

namespace hipdnn_ep_real {

// The cache maps a packed zero_points GPU pointer to its unpacked device buffer
// (and that buffer's byte size). The fields hold only void*/size_t, so the
// struct is HIP-free and embeddable by value in any op-state struct; the
// destructor (which hipFree's the cached buffers) is defined out-of-line in
// matmul_nbits.cpp so this header pulls in no HIP dependency.
struct ZpUnpackCache {
  std::unordered_map<const void *, std::pair<void *, size_t>> u8;
  std::unordered_map<const void *, std::pair<void *, size_t>> fp16;
  // Kept for correctness if concurrent Compute() on one slot ever happens
  // (today a single session drives its slots sequentially on one stream).
  std::mutex mu;

  ZpUnpackCache() = default;
  ~ZpUnpackCache();
  ZpUnpackCache(const ZpUnpackCache &) = delete;
  ZpUnpackCache &operator=(const ZpUnpackCache &) = delete;
};

// Returns the cached uint8 zero_points buffer for `zp_packed`, allocating +
// unpacking on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_unpack_zp_u8(ZpUnpackCache &cache, void *stream,
                                   const void *zp_packed, int N, int groups_k);

// Returns the cached fp16 zero_points buffer for `zp_packed`, allocating +
// converting on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_convert_zp_fp16(ZpUnpackCache &cache, void *stream,
                                      const void *zp_packed, int N,
                                      int groups_k);

// Lazily creates (on first use) and returns the per-session ZpUnpackCache owned
// by RuntimeState::zp_unpack_cache. Used by wrap_qmoe (matmul_nbits owns a
// per-instance cache in its op-state slot instead). Defined in matmul_nbits.cpp
// where the full RuntimeState definition + ZpUnpackCache HIP teardown live.
ZpUnpackCache *get_or_create_zp_cache(RuntimeState *state);

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H
