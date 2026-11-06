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

struct MatmulDerivedSizes {
  int64_t inputBatchStride;
  int64_t weightBatchStride;
  int64_t outputBatchStride;
  int64_t outputByteCount;
};

inline constexpr bool checkedPositiveProduct(int64_t lhs, int64_t rhs,
                                             int64_t &result) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

inline constexpr bool validateMatmulArguments(int64_t M, int64_t N, int64_t K,
                                              int64_t batchCount,
                                              int64_t bBatchStride,
                                              int64_t transA, int64_t transB,
                                              int64_t outputElementSize,
                                              MatmulDerivedSizes &derived) {
  if (M <= 0 || N <= 0 || K <= 0 || batchCount <= 0 || bBatchStride < 0 ||
      (transA != 0 && transA != 1) || (transB != 0 && transB != 1) ||
      outputElementSize <= 0)
    return false;

  int64_t outputElementCount = 0;
  if (!checkedPositiveProduct(M, K, derived.inputBatchStride) ||
      !checkedPositiveProduct(K, N, derived.weightBatchStride) ||
      !checkedPositiveProduct(M, N, derived.outputBatchStride) ||
      !checkedPositiveProduct(batchCount, derived.outputBatchStride,
                              outputElementCount) ||
      !checkedPositiveProduct(outputElementCount, outputElementSize,
                              derived.outputByteCount))
    return false;

  return batchCount == 1 || bBatchStride == 0 ||
         bBatchStride == derived.weightBatchStride;
}

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

// Output tiles hip_gemm_bf16_f32_transposed_b launches for an M x N result:
// 128 x 64 from M = 128 on, 64 x 64 below (keep in step with the launcher).
inline constexpr int64_t bf16GemmTileCount(int64_t M, int64_t N) {
  const int64_t tileM = M >= 128 ? 128 : 64;
  return ((M + tileM - 1) / tileM) * ((N + 63) / 64);
}

// The launcher's grid has one block row per 64 columns, and a grid
// dimension other than x holds at most 65535 blocks.
inline constexpr int64_t kBf16GemmMaxN = int64_t{65535} * 64;
// Measured on gfx1151 against hipBLASLt: with fewer output tiles than this
// the GEMM leaves most of the GPU idle and is slower (N = 64 or 256 at small
// M, up to 1.9x).
inline constexpr int64_t kBf16GemmMinTiles = 8;
// Below M = 64 a 64-row tile is mostly padding and each block walks all of
// K; past this K the GEMM is no faster than hipBLASLt (K = 8192) or slower
// (K = 14336, up to 1.25x).
inline constexpr int64_t kBf16GemmShortMMaxK = 4096;

// Prefill-sized BF16 projections take the custom WMMA GEMM instead of
// hipBLASLt on the validated target, where it has enough tiles to fill the
// GPU and, for short M, a short enough K.
inline constexpr bool useBf16F32TransposedBGemm(bool isValidatedTarget,
                                                int64_t M, int64_t N, int64_t K,
                                                int64_t batchCount,
                                                int64_t transA,
                                                int64_t transB) {
  return isValidatedTarget && M > 1 && batchCount == 1 && transA == 0 &&
         transB == 1 && M <= std::numeric_limits<int>::max() && N > 0 &&
         N <= kBf16GemmMaxN && K > 0 && K % 8 == 0 &&
         K <= std::numeric_limits<int>::max() &&
         bf16GemmTileCount(M, N) >= kBf16GemmMinTiles &&
         (M >= 64 || K <= kBf16GemmShortMMaxK);
}

inline bool hasBf16VectorAlignment(const void *A, const void *B) {
  constexpr uintptr_t kVectorAlignmentMask = 16 - 1;
  return ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
          kVectorAlignmentMask) == 0;
}

inline bool hasBf16PairAlignment(const void *A, const void *B) {
  constexpr uintptr_t kPairAlignmentMask = alignof(uint32_t) - 1;
  return ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
          kPairAlignmentMask) == 0;
}

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_MATMUL_DISPATCH_H
