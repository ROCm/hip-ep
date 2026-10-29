/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "matmul_dispatch.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

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

// gfx1151's generic fp32 GEMM kernels, keyed by solution index as hipBLASLt
// numbers them: MT8x8x8, MT16x16x8, MT8x8x16, MT16x16x16.
std::string kernelName(const FakeResult &r) {
  switch (r.algo.data[0]) {
  case 4:
    return "Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT8x8x8_SN_LDSB0";
  case 5:
    return "Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x8_SN_LDSB0";
  case 6:
    return "Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT8x8x16_SN_LDSB0";
  case 7:
    return "Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x16_SN_LDSB0";
  default:
    return "";
  }
}

bool orderIs(const FakeResult *results, std::initializer_list<int> indices) {
  int i = 0;
  for (int index : indices)
    if (results[i++].algo.data[0] != index)
      return false;
  return true;
}

bool macroTilesParse() {
  hipdnn_ep_real::MacroTile tile;
  if (!hipdnn_ep_real::parseMacroTile(
          "Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT128x64x32_SN_LDSB0_"
          "MIAV0_MO40_TT8_8_WG16_8_1",
          tile) ||
      tile.m != 128 || tile.n != 64 || tile.depth != 32)
    return false;
  // The first well-formed _MT field wins; malformed ones are skipped.
  if (!hipdnn_ep_real::parseMacroTile("x_MTx_MT8x_MT16x16x8_SN", tile) ||
      tile.m != 16 || tile.n != 16 || tile.depth != 8)
    return false;
  return !hipdnn_ep_real::parseMacroTile("", tile) &&
         !hipdnn_ep_real::parseMacroTile("Cijk_MO40_SN", tile) &&
         !hipdnn_ep_real::parseMacroTile("Cijk_MT16x16_SN", tile) &&
         !hipdnn_ep_real::parseMacroTile("Cijk_MT0x16x8_SN", tile);
}

bool unrankedCandidatesAreOrdered() {
  // Whatever order hipBLASLt returns them in, the largest output tile leads,
  // then the smaller depth, then the solution index.
  FakeResult first[] = {fake(6, 1.0f), fake(4, 1.0f), fake(7, 1.0f),
                        fake(5, 1.0f)};
  FakeResult second[] = {fake(7, 1.0f), fake(5, 1.0f), fake(4, 1.0f),
                         fake(6, 1.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(first, 4, kernelName);
  hipdnn_ep_real::orderUnrankedCandidates(second, 4, kernelName);
  if (!orderIs(first, {5, 7, 4, 6}) || !orderIs(second, {5, 7, 4, 6}))
    return false;
  // A name without a macro tile goes last; such names fall back to the index.
  FakeResult unnamed[] = {fake(9, 1.0f), fake(4, 1.0f), fake(8, 1.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(unnamed, 3, kernelName);
  if (!orderIs(unnamed, {4, 8, 9}))
    return false;
  // A ranked list keeps hipBLASLt's order, as does a list with any ranked
  // candidate.
  FakeResult ranked[] = {fake(4, 0.0f), fake(5, 0.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(ranked, 2, kernelName);
  FakeResult mixed[] = {fake(4, 1.0f), fake(5, 0.0f)};
  hipdnn_ep_real::orderUnrankedCandidates(mixed, 2, kernelName);
  return orderIs(ranked, {4, 5}) && orderIs(mixed, {4, 5});
}
} // namespace

int main() {
  if (!macroTilesParse() || !unrankedCandidatesAreOrdered())
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
