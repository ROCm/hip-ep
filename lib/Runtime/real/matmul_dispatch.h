/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_MATMUL_DISPATCH_H
#define HIPDNN_EP_MATMUL_DISPATCH_H

#include <cstdint>
#include <limits>
#include <string_view>

namespace hipdnn_ep_real {

inline bool isGfx1151(std::string_view architecture) {
  return architecture == "gfx1151" ||
         (architecture.size() >= 8 && architecture.substr(0, 8) == "gfx1151:");
}

inline constexpr bool
useF32TransposedBDecodeGemv(bool isValidatedTarget, int64_t M, int64_t N,
                            int64_t K, int64_t batchCount, int64_t elementSize,
                            int64_t transA, int64_t transB) {
  return isValidatedTarget && M == 1 && batchCount == 1 && elementSize == 4 &&
         transA == 0 && transB == 1 && N > 0 && N <= 256 && K >= 256 &&
         K <= std::numeric_limits<int>::max();
}

// The multi-wave kernel covers every decode GEMV shape the caller admits;
// K is bounded there.
inline constexpr bool useF32TransposedBMultiwaveGfx1151(int64_t N, int64_t K) {
  return N > 0 && N <= 256 && K > 0;
}

inline constexpr bool
useBf16F32TransposedBDecodeGemv(bool isValidatedTarget, int64_t M, int64_t N,
                                int64_t K, int64_t batchCount, int64_t transA,
                                int64_t transB) {
  return isValidatedTarget && M == 1 && batchCount == 1 && transA == 0 &&
         transB == 1 && N >= 1024 && N <= std::numeric_limits<int>::max() &&
         K >= 256 && K % 2 == 0 && K <= std::numeric_limits<int>::max();
}

inline bool hasBf16PairAlignment(const void *A, const void *B) {
  constexpr uintptr_t kPairAlignmentMask = alignof(uint32_t) - 1;
  return ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
          kPairAlignmentMask) == 0;
}

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_MATMUL_DISPATCH_H
