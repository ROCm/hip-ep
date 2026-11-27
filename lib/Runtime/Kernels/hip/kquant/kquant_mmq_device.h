/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The tile kernel below is ported from llama.cpp's
// ggml/src/ggml-cuda/mmq.cuh (with mmq-load-tiles.cuh, mmq-vec-dot.cuh and
// the helpers they use from mma.cuh, vecdotq.cuh and common.cuh) at commit
// 1623d8ce47bc0bee9757068780c23ca7abd45160, and its RDNA 3.5 tile
// configurations from mmq-config-rdna3-5.cuh at commit 4dd6dd4b9. llama.cpp
// is Copyright (c) 2023-2026 The ggml authors and is distributed under the
// MIT License; see lib/Runtime/Kernels/licenses/llama.cpp/LICENSE.

//===- kquant_mmq_device.h - quantized matrix tiles (private) ------===//
//
// llama.cpp's quantized matrix product (MMQ) as it runs on RDNA 3.5: weights
// stay in their K-quant blocks, activations are Q8_1 in MMQ's layout, and each
// workgroup multiplies a tile of I weight rows by J activation rows with
// 8-bit WMMA, one 256-value reduction step at a time.
//
// The port keeps llama.cpp's arithmetic and tiles, so products match its HIP
// backend: a step's weights are unpacked into shared memory as 8-bit values
// with float (or half2 scale and minimum) scales per 16 or 32 values, the
// WMMA sums are scaled and accumulated in F32 in the same order. It covers
// the formats run through MMQ - Q4_K, Q5_K and Q6_K - and
// leaves out llama.cpp's other formats and GPUs (DP4A, NVIDIA, CDNA and RDNA
// 4 paths), routed (MUL_MAT_ID) products, its stream-k decomposition and the
// bounds checks for row counts that no tile divides (the launcher picks a
// tile that divides them).

#ifndef HIP_KQUANT_MMQ_DEVICE_H
#define HIP_KQUANT_MMQ_DEVICE_H

#include "hip/kquant_abi.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// The tiles use RDNA 3.5's WMMA layout and tile configurations, and the
// launcher runs them on gfx1151 only. Device code is generated in the slices
// CMake builds with HIP_KQUANT_MMQ_DEVICE_ENABLED (gfx1151 and gfx11-generic,
// the two a gfx1151 loads), and the launcher checks the same flag: the target
// macros cannot decide it, as gfx11-generic defines none of RDNA 3.5's.
#if defined(__HIP_DEVICE_COMPILE__) && HIP_KQUANT_MMQ_DEVICE_ENABLED
#define HIP_KQUANT_MMQ_DEVICE 1
#endif

namespace kquant_mmq {

constexpr int kWarpSize = 32;
// Values of K one pass over the weights reduces (MMQ_ITER_K).
constexpr int kReductionStep = 256;
// 32-bit words of 8-bit values a tile row holds per half step
// (MMQ_TILE_NE_K): a step is two halves of 128 values.
constexpr int kTileWords = 32;
// An activation block: 128 values with their four scales (QK8_1_MMQ), and
// the 32-bit words of one in shared memory (MMQ_TILE_Y_K).
constexpr int kActivationBlockValues = 128;
constexpr int kActivationRowWords = kTileWords + 4;
constexpr int kMaxTileWidth = 128;
// The weight row tile is I = threads / 2 rows: each wave owns 16.
constexpr int kRowsPerWave = 16;

/// Activations in MMQ's layout: blocks of 128 values, `[K / 128][row]`, each
/// with a scale per 32 values (D4) or a binary16 scale and partial sum per
/// 32 values (DS4) in the first 16 bytes.
struct ActivationBlock {
  union {
    float d4[4];
    half2 ds4[4];
  };
  int8_t qs[kActivationBlockValues];
};
static_assert(sizeof(ActivationBlock) ==
                  kActivationBlockValues + 4 * sizeof(half2),
              "unexpected MMQ activation block size");
static_assert(sizeof(ActivationBlock) == 4 * HIP_Q8_1_BLOCK_SIZE,
              "an MMQ activation block is four Q8_1 blocks");

/// Values per weight block.
constexpr int kBlockValues = HIP_KQUANT_BLOCK_ELEMENTS;

/// Whether the format's activations carry partial sums (DS4): those whose
/// blocks have a minimum.
__host__ __device__ constexpr bool readsSums(int format) {
  return format == HIP_KQUANT_FORMAT_Q4_K ||
         format == HIP_KQUANT_FORMAT_Q5_K;
}

/// 32-bit words per weight row in the shared tile: 64 of 8-bit values, the
/// scales, and padding against bank conflicts (ggml_cuda_mmq_get_sram_stride).
__host__ __device__ constexpr int tileStride(int format) {
  if (format == HIP_KQUANT_FORMAT_Q6_K)
    return 2 * kTileWords + 1 + kTileWords / 8 + 7; // d and 16 8-bit scales
  return 2 * kTileWords + 2 * kTileWords / 8 + 4;   // a scale per 32 values
}

/// Threads of the workgroup that multiplies `format` weights by a tile of
/// `J` activation rows, 0 if there is none: llama.cpp's RDNA 3.5
/// configurations (all at occupancy 2). The narrow 64-row tiles are taken
/// where they leave room for a second workgroup per compute unit.
__host__ __device__ constexpr int tileThreads(int format, int J) {
  if (J != 16 && J != 32 && J != 48 && J != 64 && J != 80 && J != 96 &&
      J != 112 && J != 128)
    return 0;
  if (J <= 32)
    return tileThreads(format, 64) != 0 ? 128 : 0;
  switch (format) {
  case HIP_KQUANT_FORMAT_Q4_K:
    return J == 96 ? 256 : 128;
  case HIP_KQUANT_FORMAT_Q5_K:
    return J == 96 || J == 128 ? 256 : 128;
  case HIP_KQUANT_FORMAT_Q6_K:
    return J == 96 || J == 112 ? 256 : 128;
  default:
    return 0;
  }
}

/// Weight rows per tile.
__host__ __device__ constexpr int tileRows(int format, int J) {
  return tileThreads(format, J) / 2;
}

__host__ __device__ constexpr size_t padTo(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

/// Shared memory of one workgroup: its activation rows (padded to whole
/// passes of the workgroup's copy) and weight rows.
__host__ __device__ constexpr size_t sharedBytes(int format, int J) {
  return padTo(J * sizeof(ActivationBlock),
               tileThreads(format, J) * sizeof(int)) +
         tileRows(format, J) * tileStride(format) * sizeof(int);
}

// Helpers (vecdotq.cuh, common.cuh, vendors/hip.h).

__device__ __forceinline__ int wordOf2(const void *x, int index) {
  const uint16_t *halves = static_cast<const uint16_t *>(x);
  int word = halves[2 * index + 0] << 0;
  word |= halves[2 * index + 1] << 16;
  return word;
}

__device__ __forceinline__ int wordOf4(const void *x, int index) {
  return static_cast<const int *>(x)[index];
}

__device__ __forceinline__ float halfToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(__half(raw));
}

/// Two adjacent binary16 fields read as one pair.
__device__ __forceinline__ half2 halfPair(const uint16_t *first) {
  return *reinterpret_cast<const half2 *>(first);
}

/// Bytewise a - b, wrapping.
__device__ __forceinline__ int subtractBytes(int a, int b) {
  const unsigned aLarge = a | 0x80808080;
  const unsigned bSmall = b & 0x7f7f7f7f;
  const unsigned lowBits = aLarge - bSmall;
  const unsigned flipHighBit = (a ^ ~b) & 0x80808080;
  return lowBits ^ flipHighBit;
}

// RDNA 3 WMMA fragments (mma.cuh). An operand fragment of 16 rows by W
// 32-bit words of 8-bit values: each lane holds all of row lane % 16 (both
// halves of the wave hold the same rows). The 16 x 16 result: lane holds
// column lane % 16 of rows 2 * l + lane / 16.

template <int Words> struct Operand {
  int x[Words];
};

/// Row `lane % 16` of a 16-row operand from `stride`-word rows.
template <int Words>
__device__ __forceinline__ void loadOperand(Operand<Words> &t, const int *rows,
                                            int stride) {
  const int *row = rows + (threadIdx.x % 16) * stride;
  if constexpr (Words == 8) {
    *reinterpret_cast<int4 *>(t.x + 0) =
        *reinterpret_cast<const int4 *>(row + 0);
    *reinterpret_cast<int4 *>(t.x + 4) =
        *reinterpret_cast<const int4 *>(row + 4);
  } else {
    static_assert(Words == 4, "operands are 16 or 32 values wide");
    *reinterpret_cast<int2 *>(t.x + 0) =
        *reinterpret_cast<const int2 *>(row + 0);
    *reinterpret_cast<int2 *>(t.x + 2) =
        *reinterpret_cast<const int2 *>(row + 2);
  }
}

struct Result {
  int x[8] = {0};
  /// The weight row (i) and activation row (j) of element l.
  static __device__ __forceinline__ int row(int l) {
    return 2 * l + threadIdx.x / 16;
  }
  static __device__ __forceinline__ int column() { return threadIdx.x % 16; }
};

/// D = A B^T over 32 (two WMMAs) or 16 values of K.
template <int Words>
__device__ __forceinline__ void multiply(Result &d, const Operand<Words> &a,
                                         const Operand<Words> &b) {
#if defined(HIP_KQUANT_MMQ_DEVICE)
  using int32x8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;
  using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  int32x8 *acc = reinterpret_cast<int32x8 *>(d.x);
  const int32x4 *aVec = reinterpret_cast<const int32x4 *>(a.x);
  const int32x4 *bVec = reinterpret_cast<const int32x4 *>(b.x);
  if constexpr (Words == 8) {
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, aVec[0], true,
                                                        bVec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, aVec[1], true,
                                                        bVec[1], acc[0], true);
  } else {
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, aVec[0], true,
                                                        bVec[0], acc[0], false);
  }
#endif
}

// Weight tile loads (mmq-load-tiles.cuh): one 256-value step of rows
// [0, I) from block `block0` of row 0, rows `rowBlocks` blocks apart, into
// `tile` (`Stride` words per row: 64 words of 8-bit values, then scales).

/// Four 6-bit Q4_K/Q5_K scales (ksc 0, 1) or minimums (ksc 2, 3), one per
/// byte.
__device__ __forceinline__ int unpackScalesQ45K(const int *scales, int ksc) {
  return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) &
          0x0F0F0F0F) |
         ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);
}

/// The scale and negated minimum of each 32 values of a Q4_K/Q5_K row.
template <typename Block, int I, int Warps, int Stride, bool Checked>
__device__ __forceinline__ void loadScalesQ45K(const char *__restrict__ x,
                                               half2 *__restrict__ dm,
                                               int block0, int rowBlocks) {
  constexpr int rowsPerWarp = kWarpSize / 2;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps * rowsPerWarp) {
    // As in llama.cpp, Q4_K bounds the row and Q5_K wraps it; with I a
    // multiple of Warps * 16 neither changes it.
    int i = i0 + threadIdx.y * rowsPerWarp + threadIdx.x / 2;
    if constexpr (!Checked)
      i %= I;
    if (!Checked || i < I) {
      const Block *bxi =
          reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
      const int *scales = reinterpret_cast<const int *>(bxi->scales);
      const int ksc = threadIdx.x % 2;
      const int sc32 = unpackScalesQ45K(scales, ksc + 0);
      const int m32 = unpackScalesQ45K(scales, ksc + 2);
      const uint8_t *sc8 = reinterpret_cast<const uint8_t *>(&sc32);
      const uint8_t *m8 = reinterpret_cast<const uint8_t *>(&m32);
      const half2 blockDm = halfPair(&bxi->d) * make_half2(1.0f, -1.0f);
#pragma unroll
      for (int l = 0; l < int(sizeof(int)); ++l)
        dm[i * Stride + sizeof(int) * ksc + l] =
            blockDm * make_half2(sc8[l], m8[l]);
    }
  }
}

template <int I, int Warps, int Stride>
__device__ __forceinline__ void loadQ4K(const char *__restrict__ x,
                                        int *__restrict__ tile, int block0,
                                        int rowBlocks) {
  using Block = hip_kquant_q4_k_block_t;
  int *qs = tile;
  half2 *dm = reinterpret_cast<half2 *>(qs + 2 * kTileWords);
  const int txi = threadIdx.x;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps) {
    const int i = i0 + threadIdx.y;
    const Block *bxi =
        reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
    const int qs0 = wordOf4(bxi->qs, txi);
    qs[i * Stride + 16 * (txi / 8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
    qs[i * Stride + 16 * (txi / 8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
  }
  loadScalesQ45K<Block, I, Warps, Stride, true>(x, dm, block0, rowBlocks);
}

template <int I, int Warps, int Stride>
__device__ __forceinline__ void loadQ5K(const char *__restrict__ x,
                                        int *__restrict__ tile, int block0,
                                        int rowBlocks) {
  using Block = hip_kquant_q5_k_block_t;
  int *qs = tile;
  half2 *dm = reinterpret_cast<half2 *>(qs + 2 * kTileWords);
  const int txi = threadIdx.x;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps) {
    const int i = i0 + threadIdx.y;
    const Block *bxi =
        reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
    const int ky = 2 * txi;
    const int ql = wordOf4(bxi->qs, txi);
    const int ql0 = (ql >> 0) & 0x0F0F0F0F;
    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
    const int qh = wordOf4(bxi->qh, txi % 8);
    const int qh0 = ((qh >> (2 * (txi / 8) + 0)) << 4) & 0x10101010;
    const int qh1 = ((qh >> (2 * (txi / 8) + 1)) << 4) & 0x10101010;
    const int kq0 = ky - ky % 16 + txi % 8 + 0;
    const int kq1 = ky - ky % 16 + txi % 8 + 8;
    qs[i * Stride + kq0] = ql0 | qh0;
    qs[i * Stride + kq1] = ql1 | qh1;
  }
  loadScalesQ45K<Block, I, Warps, Stride, false>(x, dm, block0, rowBlocks);
}

template <int I, int Warps, int Stride>
__device__ __forceinline__ void loadQ6K(const char *__restrict__ x,
                                        int *__restrict__ tile, int block0,
                                        int rowBlocks) {
  using Block = hip_kquant_q6_k_block_t;
  int *qs = tile;
  float *df = reinterpret_cast<float *>(qs + 2 * kTileWords);
  int *sc = reinterpret_cast<int *>(df + 1);
  const int txi = threadIdx.x;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps) {
    const int i = i0 + threadIdx.y;
    const Block *bxi =
        reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
    const int ql = wordOf2(bxi->ql, txi);
    const int ql0 = (ql >> 0) & 0x0F0F0F0F;
    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
    const int qh = wordOf2(bxi->qh, 8 * (txi / 16) + txi % 8);
    const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
    const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;
    const int kq0 = 2 * txi - txi % 16 + 0;
    const int kq1 = 2 * txi - txi % 16 + 16;
    qs[i * Stride + kq0] = subtractBytes(ql0 | qh0, 0x20202020);
    qs[i * Stride + kq1] = subtractBytes(ql1 | qh1, 0x20202020);
  }

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps * kWarpSize) {
    const int i = (i0 + threadIdx.y * kWarpSize + threadIdx.x) % I;
    const Block *bxi =
        reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
    df[i * Stride] = halfToFloat(bxi->d);
  }

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += Warps * 8) {
    const int i = (i0 + threadIdx.y * 8 + threadIdx.x / 4) % I;
    const Block *bxi =
        reinterpret_cast<const Block *>(x) + block0 + i * rowBlocks;
    sc[i * Stride + threadIdx.x % 4] = wordOf2(bxi->scales, threadIdx.x % 4);
  }
}

template <int Format, int I, int Warps>
__device__ __forceinline__ void loadWeights(const char *__restrict__ x,
                                            int *__restrict__ tile, int block0,
                                            int rowBlocks) {
  constexpr int S = tileStride(Format);
  if constexpr (Format == HIP_KQUANT_FORMAT_Q4_K)
    loadQ4K<I, Warps, S>(x, tile, block0, rowBlocks);
  else if constexpr (Format == HIP_KQUANT_FORMAT_Q5_K)
    loadQ5K<I, Warps, S>(x, tile, block0, rowBlocks);
  else if constexpr (Format == HIP_KQUANT_FORMAT_Q6_K)
    loadQ6K<I, Warps, S>(x, tile, block0, rowBlocks);
  else
    static_assert(Format < 0, "no MMQ tile load for this format");
}

// Tile products (mmq-vec-dot.cuh): adds the product of each wave's 16
// weight rows with the J activation rows over one half step (128 values from
// word `k00`) to `sum`, element l of the 16-column block j0 / 16 at
// sum[(j0 / 16) * 8 + l].

enum class Scales {
  Per32WithMin, // a half2 scale and minimum per 32 values (DS4 activations)
  Q6K,          // a float per row and an 8-bit scale per 16 values
};

__host__ __device__ constexpr Scales scalesOf(int format) {
  return readsSums(format) ? Scales::Per32WithMin : Scales::Q6K;
}

template <int Format, int J>
__device__ __forceinline__ void
multiplyTiles(const int *__restrict__ x, const int *__restrict__ y,
              float *__restrict__ sum, int k00) {
  constexpr int S = tileStride(Format);
  constexpr Scales scales = scalesOf(Format);
  // 32 values (two WMMAs) per product, or 16 where scales change every 16.
  constexpr int words = scales == Scales::Q6K ? 4 : 8;

  const int *xQs = x;
  const float *xDf = reinterpret_cast<const float *>(xQs) + 2 * kTileWords;
  const half2 *xDm = reinterpret_cast<const half2 *>(xQs) + 2 * kTileWords;
  const int *xSc = reinterpret_cast<const int *>(xDf) + 1;
  const int *yQs = y + 4;
  const float *yDf = reinterpret_cast<const float *>(y);
  const half2 *yDm = reinterpret_cast<const half2 *>(y);

  const int i0 = threadIdx.y * kRowsPerWave;

  for (int k01 = 0; k01 < kTileWords; k01 += words) {
    const int k0 = k00 + k01;

    Operand<words> a;
    loadOperand(a, xQs + i0 * S + k0, S);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += 16) {
      Operand<words> b;
      loadOperand(b, yQs + j0 * kActivationRowWords + k01, kActivationRowWords);
      const int j = j0 + Result::column();

      Result c;
      multiply(c, a, b);

      if constexpr (scales == Scales::Per32WithMin) {
        const float2 dsB =
            __half22float2(yDm[j * kActivationRowWords + k01 / 8]);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int i = i0 + Result::row(l);
          const float2 dmA = __half22float2(xDm[i * S + k0 / 8]);
          sum[(j0 / 16) * 8 + l] += dmA.x * dsB.x * c.x[l];
          sum[(j0 / 16) * 8 + l] += dmA.y * dsB.y;
        }
      } else {
        const float dB = yDf[j * kActivationRowWords + k01 / 8];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int i = i0 + Result::row(l);
          const int8_t *sc =
              reinterpret_cast<const int8_t *>(xSc + i * S + k00 / 16);
          sum[(j0 / 16) * 8 + l] += c.x[l] * sc[k01 / 4] * xDf[i * S] * dB;
        }
      }
    }
  }
}

/// The weight tile multiplied by J activation rows (blockIdx.x's I weight
/// rows, blockIdx.y's activation rows, blockIdx.z's sample). Activations are
/// `[sample][K / 128][column]` MMQ blocks.
template <int Format, int J>
__global__ void __launch_bounds__(tileThreads(Format, J), 2)
    mmqKernel(const char *__restrict__ x, const int *__restrict__ y,
              float *__restrict__ dst, const int outputColumns,
              const int rowBlocks, const int columns, const int outputStride,
              const int stepBlocks, const int activationSampleStride,
              const int outputSampleStride) {
#if defined(HIP_KQUANT_MMQ_DEVICE)
  constexpr int threads = tileThreads(Format, J);
  constexpr int warps = threads / kWarpSize;
  constexpr int I = tileRows(Format, J);
  static_assert(threads != 0 && I == warps * kRowsPerWave);
  const int thread = threadIdx.y * kWarpSize + threadIdx.x;

  // The tile's activation rows and its weight rows.
  extern __shared__ int shared[];
  int *tileY = shared;
  int *tileX = tileY + padTo(J * kActivationRowWords, threads);

  const int sample = blockIdx.z;
  const int jt = blockIdx.y;
  const int it = blockIdx.x;

  const int offsetY = sample * activationSampleStride +
                      jt * J * (sizeof(ActivationBlock) / sizeof(int));
  const int offsetDst =
      sample * outputSampleStride + jt * J * outputStride + it * I;
  const int lastColumn = outputColumns - jt * J - 1;
  const int offsetX = it * I * rowBlocks;

  const int *yTile = y + offsetY;
  constexpr int blocksPerStep = kReductionStep / kBlockValues;
  constexpr int activationWords = sizeof(ActivationBlock) / sizeof(int);
  float sum[J * I / threads] = {0.0f};

  for (int kb0 = 0; kb0 < stepBlocks; kb0 += blocksPerStep) {
    loadWeights<Format, I, warps>(x, tileX, offsetX + kb0, rowBlocks);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int *by0 =
          yTile + columns * ((kb0 * kBlockValues / kActivationBlockValues) *
                                 activationWords +
                             half * activationWords);
#pragma unroll
      for (int l0 = 0; l0 < J * kActivationRowWords; l0 += threads)
        tileY[l0 + thread] = by0[l0 + thread];
      __syncthreads();
      multiplyTiles<Format, J>(tileX, tileY, sum, half * kTileWords);
      __syncthreads();
    }
  }

  // Each wave writes its 16 rows of the J columns.
  float *out = dst + offsetDst;
  const int i0 = threadIdx.y * kRowsPerWave;
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += 16) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const int j = j0 + Result::column();
      if (j > lastColumn)
        continue;
      const int i = i0 + Result::row(l);
      out[j * outputStride + i] = sum[(j0 / 16) * 8 + l];
    }
  }
#endif
}

} // namespace kquant_mmq

#endif // HIP_KQUANT_MMQ_DEVICE_H
