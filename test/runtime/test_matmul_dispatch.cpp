/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "matmul_dispatch.h"

#include <cstdint>
#include <limits>

using hipdnn_ep_real::useF32TransposedBDecodeGemv;
using hipdnn_ep_real::useF32TransposedBMultiwaveGfx1151;

namespace {
// The fields of hipblasLtMatmulHeuristicResult_t the ordering reads.
struct FakeResult {
  struct {
    uint8_t data[16];
  } algo;
  float wavesCount;
};

FakeResult fake(uint8_t index, float waves) {
  FakeResult r{};
  r.algo.data[0] = index;
  r.wavesCount = waves;
  return r;
}

bool unrankedCandidatesAreOrdered() {
  FakeResult unranked[] = {fake(6, 1.0f), fake(4, 1.0f), fake(7, 1.0f),
                           fake(5, 1.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(unranked, 4);
  for (int i = 0; i < 4; ++i)
    if (unranked[i].algo.data[0] != 4 + i)
      return false;
  // A ranked list keeps hipBLASLt's order.
  FakeResult ranked[] = {fake(9, 0.0f), fake(3, 0.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(ranked, 2);
  return ranked[0].algo.data[0] == 9 && ranked[1].algo.data[0] == 3;
}
} // namespace

int main() {
  if (!unrankedCandidatesAreOrdered())
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
