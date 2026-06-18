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
// Ownership (see docs/design/op-state-slots-design.md): this cache is now
// per-op-instance. matmul_nbits owns one in its MatmulNbitsState op-state slot;
// qmoe embeds one in its QmoeState slot. It used to be a single shared
// RuntimeState::zp_unpack_cache, which two concurrent sessions could contend
// on. Each op instance's zero_points pointers are distinct, so a per-instance
// cache holds exactly that instance's entries (matmul_nbits: 1 pointer; qmoe:
// one per expert) with no cross-instance/cross-session sharing.

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>

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

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_REAL_ZP_UNPACK_CACHE_H
