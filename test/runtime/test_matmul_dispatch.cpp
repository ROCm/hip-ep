/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "matmul_dispatch.h"

#include <cstdint>
#include <limits>

using hipdnn_ep_real::MatmulDerivedSizes;
using hipdnn_ep_real::useBf16F32TransposedBDecodeGemv;
using hipdnn_ep_real::useF32TransposedBDecodeGemv;
using hipdnn_ep_real::useF32TransposedBMultiwaveGfx1151;
using hipdnn_ep_real::validateMatmulArguments;

int main() {
  MatmulDerivedSizes derived{};
  if (!validateMatmulArguments(/*M=*/2, /*N=*/3, /*K=*/5,
                               /*batchCount=*/4, /*bBatchStride=*/15,
                               /*transA=*/0, /*transB=*/1,
                               /*outputElementSize=*/4, derived) ||
      derived.inputBatchStride != 10 || derived.outputBatchStride != 6 ||
      derived.weightBatchStride != 15 || derived.outputByteCount != 96)
    return 1;

  if (!validateMatmulArguments(/*M=*/2, /*N=*/3, /*K=*/5,
                               /*batchCount=*/4, /*bBatchStride=*/0,
                               /*transA=*/0, /*transB=*/1,
                               /*outputElementSize=*/4, derived) ||
      derived.weightBatchStride != 15)
    return 1;

  const auto rejects = [](int64_t M, int64_t N, int64_t K, int64_t batchCount,
                          int64_t bBatchStride, int64_t transA, int64_t transB,
                          int64_t outputElementSize) {
    MatmulDerivedSizes ignored{};
    return !validateMatmulArguments(M, N, K, batchCount, bBatchStride, transA,
                                    transB, outputElementSize, ignored);
  };
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (!rejects(0, 3, 5, 4, 15, 0, 1, 4) || !rejects(2, 0, 5, 4, 15, 0, 1, 4) ||
      !rejects(2, 3, 0, 4, 15, 0, 1, 4) || !rejects(2, 3, 5, 0, 15, 0, 1, 4) ||
      !rejects(-2, 3, 5, 4, 15, 0, 1, 4) ||
      !rejects(2, -3, 5, 4, 15, 0, 1, 4) ||
      !rejects(2, 3, -5, 4, 15, 0, 1, 4) ||
      !rejects(2, 3, 5, -4, 15, 0, 1, 4) || !rejects(2, 3, 5, 4, -1, 0, 1, 4) ||
      !rejects(2, 3, 5, 4, 14, 0, 1, 4) || !rejects(2, 3, 5, 4, 15, -1, 1, 4) ||
      !rejects(2, 3, 5, 4, 15, 0, 2, 4) ||
      !rejects(kMax, 1, 2, 1, 0, 0, 1, 4) ||
      !rejects(kMax, 2, 1, 1, 0, 0, 1, 4) ||
      !rejects(1, 5, int64_t{1} << 62, 2, int64_t{1} << 62, 0, 1, 4) ||
      !rejects(2, 2, 1, kMax, 0, 0, 1, 4) ||
      !rejects(1, kMax / 4 + 1, 1, 1, 0, 0, 1, 4))
    return 1;

  alignas(uint32_t) uint8_t storage[12] = {};
  if (!hipdnn_ep_real::hasBf16PairAlignment(storage, storage + 4) ||
      hipdnn_ep_real::hasBf16PairAlignment(storage + 2, storage + 4) ||
      hipdnn_ep_real::hasBf16PairAlignment(storage, storage + 6))
    return 1;

  if (!useF32TransposedBDecodeGemv(true, 1, 1, 256, 1, 4, 0, 1) ||
      !useF32TransposedBDecodeGemv(true, 1, 256, 4096, 1, 4, 0, 1))
    return 1;

  if (!useF32TransposedBMultiwaveGfx1151(1, 2048) ||
      !useF32TransposedBMultiwaveGfx1151(128, 2048) ||
      !useF32TransposedBMultiwaveGfx1151(256, 2048) ||
      !useF32TransposedBMultiwaveGfx1151(128, 2816) ||
      useF32TransposedBMultiwaveGfx1151(0, 2048) ||
      useF32TransposedBMultiwaveGfx1151(257, 2048) ||
      !useF32TransposedBMultiwaveGfx1151(128, 1024) ||
      !useF32TransposedBMultiwaveGfx1151(128, 2880) ||
      !useF32TransposedBMultiwaveGfx1151(256, 4096) ||
      useF32TransposedBMultiwaveGfx1151(128, 0))
    return 1;

  if (!useBf16F32TransposedBDecodeGemv(true, 1, 1024, 3072, 1, 0, 1) ||
      !useBf16F32TransposedBDecodeGemv(true, 1, 128256, 3072, 1, 0, 1))
    return 1;

  if (useBf16F32TransposedBDecodeGemv(false, 1, 1024, 3072, 1, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 2, 1024, 3072, 1, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1024, 3072, 2, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1024, 3072, 1, 1, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1024, 3072, 1, 0, 0) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1023, 3072, 1, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1024, 255, 1, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(true, 1, 1024, 3073, 1, 0, 1) ||
      useBf16F32TransposedBDecodeGemv(
          true, 1, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1,
          3072, 1, 0, 1))
    return 1;

  // Prefill, batched, F16, untransposed, wide, and unrepresentable dimensions
  // must continue through the existing hipBLASLt path.
  if (useF32TransposedBDecodeGemv(false, 1, 128, 4096, 1, 4, 0, 1) ||
      useF32TransposedBDecodeGemv(true, 2, 128, 4096, 1, 4, 0, 1) ||
      useF32TransposedBDecodeGemv(true, 1, 128, 4096, 2, 4, 0, 1) ||
      useF32TransposedBDecodeGemv(true, 1, 128, 4096, 1, 2, 0, 1) ||
      useF32TransposedBDecodeGemv(true, 1, 128, 4096, 1, 4, 0, 0) ||
      useF32TransposedBDecodeGemv(true, 1, 257, 4096, 1, 4, 0, 1) ||
      useF32TransposedBDecodeGemv(
          true, 1, 128,
          static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1, 4, 0,
          1))
    return 1;

  if (!hipdnn_ep_real::isGfx1151("gfx1151") ||
      !hipdnn_ep_real::isGfx1151("gfx1151:") ||
      !hipdnn_ep_real::isGfx1151("gfx1151:sramecc+:xnack-") ||
      hipdnn_ep_real::isGfx1151("gfx11510") ||
      hipdnn_ep_real::isGfx1151("gfx1100") || hipdnn_ep_real::isGfx1151(""))
    return 1;

  return 0;
}
