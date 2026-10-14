/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "matmul_dispatch.h"

#include <cstdint>
#include <limits>

using hipdnn_ep_real::useBf16F32TransposedBDecodeGemv;
using hipdnn_ep_real::useF32TransposedBDecodeGemv;
using hipdnn_ep_real::useF32TransposedBMultiwaveGfx1151;

int main() {
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
