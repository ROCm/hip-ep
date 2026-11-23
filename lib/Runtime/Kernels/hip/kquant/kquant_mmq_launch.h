/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- kquant_mmq_launch.h - MMQ tile launch (private) -------------===//
//
// Launches the MMQ tile kernel (kquant_mmq_device.h) for one weight
// format, picking the tile width over activation rows as llama.cpp does on
// RDNA 3.5.

#ifndef HIP_KQUANT_MMQ_LAUNCH_H
#define HIP_KQUANT_MMQ_LAUNCH_H

#include "kquant_mmq_device.h"

#include <cstdint>
#include <cstdio>

namespace kquant_mmq {

/// One MMQ product. Activations are Q8_1 in the MMQ layout: blocks of 128
/// values, `[sample][K / 128][column]`; the weights are shared across
/// samples.
struct MmqProblem {
  const void *weights = nullptr;
  const int *activation = nullptr;
  float *output = nullptr;
  int rows = 0;    // N, output features
  int K = 0;       // reduction length, a multiple of 256
  int columns = 0; // activation columns per sample (M)
  int samples = 1;
  int64_t activationSampleStride = 0; // ints
  int64_t outputSampleStride = 0;     // floats
};

/// The candidate tile widths over activation columns: llama.cpp's on RDNA
/// 3.5 (its granularity: multiples of 16 from 48). A narrower tile that
/// still covers the columns in as few tiles computes fewer empty ones - 107
/// prompt rows take 112, not 128.
inline constexpr int kTileWidths[] = {16, 32, 48, 64, 80, 96, 112, 128};

template <int Format, int J>
int launchMmqTile(hipStream_t stream, const MmqProblem &p) {
  constexpr int threads = tileThreads(Format, J);
  if constexpr (threads == 0)
    return -1; // pickTileWidth never picks it
  else {
    constexpr int I = tileRows(Format, J);
    constexpr size_t bytes = sharedBytes(Format, J);
    static bool limitRaised = false;
    if (!limitRaised) {
      if (hipFuncSetAttribute(
              reinterpret_cast<const void *>(&mmqKernel<Format, J>),
              hipFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(bytes)) != hipSuccess)
        return -1;
      limitRaised = true;
    }
    const unsigned gridX = static_cast<unsigned>(p.rows / I);
    const unsigned columnTiles = static_cast<unsigned>((p.columns + J - 1) / J);
    const unsigned gridZ = static_cast<unsigned>(p.samples);
    mmqKernel<Format, J>
        <<<dim3(gridX, columnTiles, gridZ),
           dim3(kWarpSize, threads / kWarpSize), bytes, stream>>>(
            static_cast<const char *>(p.weights), p.activation, p.output,
            p.columns, p.K / kBlockValues, p.columns, p.rows,
            p.K / kBlockValues, static_cast<int>(p.activationSampleStride),
            static_cast<int>(p.outputSampleStride));
    return static_cast<int>(hipGetLastError());
  }
}

/// The narrowest available tile width that covers `columns` in the fewest
/// tiles and whose row tile divides `rows`; 0 if none does.
template <int Format> int pickTileWidth(int columns, int rows) {
  int best = 0;
  int bestTiles = 0;
  for (const int J : kTileWidths) {
    const int I = tileRows(Format, J);
    if (I == 0 || rows % I != 0)
      continue;
    const int tiles = (columns + J - 1) / J;
    if (best == 0 || tiles < bestTiles) {
      best = J;
      bestTiles = tiles;
    }
  }
  return best;
}

template <int Format>
int launchMmqFormat(hipStream_t stream, const MmqProblem &p) {
  switch (pickTileWidth<Format>(p.columns, p.rows)) {
  case 16:
    return launchMmqTile<Format, 16>(stream, p);
  case 32:
    return launchMmqTile<Format, 32>(stream, p);
  case 48:
    return launchMmqTile<Format, 48>(stream, p);
  case 64:
    return launchMmqTile<Format, 64>(stream, p);
  case 80:
    return launchMmqTile<Format, 80>(stream, p);
  case 96:
    return launchMmqTile<Format, 96>(stream, p);
  case 112:
    return launchMmqTile<Format, 112>(stream, p);
  case 128:
    return launchMmqTile<Format, 128>(stream, p);
  default:
    return -1;
  }
}

} // namespace kquant_mmq

#endif // HIP_KQUANT_MMQ_LAUNCH_H
