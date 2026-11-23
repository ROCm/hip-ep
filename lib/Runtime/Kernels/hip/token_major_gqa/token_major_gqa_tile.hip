/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * The tile kernel below is ported from llama.cpp's
 * ggml/src/ggml-cuda/fattn-tile.cuh (and the helpers it uses from common.cuh
 * and fattn-common.cuh) at commit 1623d8ce47bc0bee9757068780c23ca7abd45160.
 * llama.cpp is Copyright (c) 2023-2026 The ggml authors and is distributed
 * under the MIT License; see lib/Runtime/Kernels/licenses/llama.cpp/LICENSE.
 */

// Grouped-query attention over an F16 KV cache with llama.cpp's tile
// algorithm.
//
// The port keeps llama.cpp's arithmetic so results match its HIP backend: the
// scaled query and the running weighted-V sum are F16, scores and softmax
// statistics F32, with the same tile configurations for RDNA and other AMD
// GPUs. It covers what hip.token_major_gqa uses - an additive mask - and leaves
// out llama.cpp's NVIDIA and FP32 paths, attention sinks, logit softcapping,
// ALiBi and asymmetric K/V widths.
//
// llama.cpp sizes each launch from the KV length it knows on the host. A
// compiled decoder's cache is a fixed capacity and its live length is only
// known on the device, so the launcher sizes the grid from the capacity and
// hands the kernel the live extent through its kvMax operand, which a small
// kernel computes first from the rows the cache update wrote (or, without
// them, from the mask). Blocks past the extent do no work, and the partial
// results are merged by a combine kernel that, unlike llama.cpp's, returns
// zeros for a row whose every key is masked.
//
// Grid shape, parallel-block count and kernel configuration are functions of
// the static shape only, so a step that replays as a HIP graph stays valid as
// the live length grows.

#include "token_major_gqa_tile.h"
#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// llama.cpp's RDNA (vendors/hip.h) and V_DOT2_F32_F16_AVAILABLE
// (common.cuh) device groupings.
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1010__) ||        \
    defined(__gfx1012__) || defined(__gfx1030__) || defined(__gfx1031__) ||    \
    defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) ||    \
    defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define HIP_TOKEN_MAJOR_GQA_TILE_RDNA 1
#endif
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1030__) ||        \
    defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) ||    \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) ||    \
    defined(__gfx1037__) || defined(__gfx906__) || defined(__gfx908__) ||      \
    defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
#define HIP_TOKEN_MAJOR_GQA_TILE_DOT2 1
#endif

namespace token_major_gqa {
namespace {

constexpr int kWarpSize = 32;
constexpr int kExtentThreads = 256;
constexpr int kExtentKeysPerBlock = 16 * kExtentThreads;
constexpr int kMaximumParallelBlocks = 64;
// Widest single load or store (ggml_cuda_get_max_cpy_bytes) and the 32-bit
// words it moves.
constexpr int kCopyBytes = 16;
constexpr int kCopyWords = kCopyBytes / 4;
// Key tiles the grouped (unchecked) form needs whole: FATTN_KQ_STRIDE.
constexpr int kKqStride = 256;
// The most tokens a step may have to group padded heads (visitShape).
constexpr int kPaddedGroupTokens = 8;
// Head-room kept above the running maximum so F16 probabilities cannot
// overflow (FATTN_KQ_MAX_OFFSET).
constexpr float kKqMaxOffset = 3.0f * 0.6931f;

//===----------------------------------------------------------------------===//
// Configuration (ggml_cuda_fattn_tile_get_config_amd{,_rdna}) for the head
// dimensions instantiated here.
//===----------------------------------------------------------------------===//

struct TileConfig {
  int nthreads;
  int occupancy;
  int nbatchFa; // key rows per iteration
  int nbatchK;  // key columns loaded at once for the KQ product
};

__host__ __device__ constexpr TileConfig tileConfig(bool rdna, int d,
                                                    int ncols) {
  // clang-format off
#define CASE(D_, NCOLS, T, O, FA, K) if (d == (D_) && ncols == (NCOLS)) return TileConfig{T, O, FA, K};
  if (rdna) {
    CASE( 64,  2,  64, 8,  32,  64) CASE( 64,  4,  64, 8,  32,  64)
    CASE( 64,  8, 128, 5, 128,  64) CASE( 64, 16, 128, 5, 128,  64)
    CASE( 64, 32, 128, 4,  64,  64) CASE( 64, 64, 128, 5,  64,  64)
    CASE(128,  2,  64, 8,  32,  64) CASE(128,  4, 128, 8,  64,  64)
    CASE(128,  8, 128, 8,  64,  64) CASE(128, 16, 256, 3, 128, 128)
    CASE(128, 32, 256, 3, 128,  64) CASE(128, 64, 256, 3,  64,  64)
    CASE(256,  2,  64, 8,  32,  64) CASE(256,  4, 128, 6,  32, 256)
    CASE(256,  8, 128, 6,  32, 256) CASE(256, 16, 256, 5,  32, 256)
    CASE(256, 32, 256, 3,  64, 128)
  } else {
    CASE( 64,  2,  64, 3,  32,  64) CASE( 64,  4, 128, 3,  64,  64)
    CASE( 64,  8, 128, 2,  32,  64) CASE( 64, 16, 256, 2, 128,  64)
    CASE( 64, 32, 256, 2,  64,  64) CASE( 64, 64, 256, 2,  64,  64)
    CASE(128,  2, 256, 2, 128,  64) CASE(128,  4, 128, 2,  64, 128)
    CASE(128,  8, 256, 2,  64, 128) CASE(128, 16, 256, 2,  64, 128)
    CASE(128, 32, 256, 2,  64,  64) CASE(128, 64, 256, 2,  64,  32)
    CASE(256,  2, 256, 2, 128,  64) CASE(256,  4, 256, 2,  64, 128)
    CASE(256,  8, 256, 2,  64, 128) CASE(256, 16, 256, 2,  32, 128)
    CASE(256, 32, 256, 2,  32, 128)
  }
  // 40, 72, 80, 96 and 112 use the same table on every AMD GPU.
  CASE( 40,  2,  64, 2,  32,  40) CASE( 40,  4, 128, 2,  32,  40)
  CASE( 40,  8, 256, 2,  32,  40) CASE( 40, 16, 256, 2,  32,  40)
  CASE( 40, 32, 256, 2,  32,  40) CASE( 40, 64, 256, 2,  32,  40)
  CASE( 72,  2,  64, 2,  32,  72) CASE( 72,  4, 128, 2,  32,  72)
  CASE( 72,  8, 256, 2,  32,  72) CASE( 72, 16, 256, 2,  32,  72)
  CASE( 72, 32, 256, 2,  32,  72) CASE( 72, 64, 256, 2,  32,  72)
  CASE( 80,  2,  64, 2,  32,  40) CASE( 80,  4, 128, 2,  32,  40)
  CASE( 80,  8, 256, 2,  32,  40) CASE( 80, 16, 256, 2,  32,  40)
  CASE( 80, 32, 256, 2,  32,  40) CASE( 80, 64, 256, 2,  32,  40)
  CASE( 96,  2,  64, 2,  32,  48) CASE( 96,  4, 128, 2,  32,  48)
  CASE( 96,  8, 256, 2,  32,  48) CASE( 96, 16, 256, 2,  32,  48)
  CASE( 96, 32, 256, 2,  32,  48) CASE( 96, 64, 256, 2,  32,  48)
  CASE(112,  2,  64, 2,  32,  56) CASE(112,  4, 128, 2,  32,  56)
  CASE(112,  8, 256, 2,  32,  56) CASE(112, 16, 256, 2,  32,  56)
  CASE(112, 32, 256, 2,  32,  56) CASE(112, 64, 256, 2,  32,  56)
#undef CASE
      // clang-format on
      return TileConfig{0, 0, 0, 0};
}

#if defined(HIP_TOKEN_MAJOR_GQA_TILE_RDNA)
constexpr bool kDeviceRdna = true;
#else
constexpr bool kDeviceRdna = false;
#endif

//===----------------------------------------------------------------------===//
// Device helpers (common.cuh)
//===----------------------------------------------------------------------===//

// ggml_cuda_unroll: unrolls through recursion where a loop with an early
// return would not unroll.
template <int n> struct Unroll {
  template <typename F> __device__ void operator()(const F &f) const {
    f(n - 1);
    Unroll<n - 1>{}(f);
  }
};
template <> struct Unroll<1> {
  template <typename F> __device__ void operator()(const F &f) const { f(0); }
};

// ggml_cuda_memcpy_1: one load and store of `bytes`, between registers and
// shared or global memory.
template <int bytes>
__device__ __forceinline__ void copyOnce(void *__restrict__ dst,
                                         const void *__restrict__ src) {
  if constexpr (bytes == 2)
    *static_cast<short *>(dst) = *static_cast<const short *>(src);
  else if constexpr (bytes == 4)
    *static_cast<int *>(dst) = *static_cast<const int *>(src);
  else if constexpr (bytes == 8)
    *static_cast<int2 *>(dst) = *static_cast<const int2 *>(src);
  else if constexpr (bytes == 16)
    *static_cast<int4 *>(dst) = *static_cast<const int4 *>(src);
  else
    static_assert(bytes == 2, "unsupported copy width");
}

template <int width = kWarpSize>
__device__ __forceinline__ float warpReduceSum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1)
    x += __shfl_xor(x, offset, width);
  return x;
}

template <int width = kWarpSize>
__device__ __forceinline__ float warpReduceMax(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1)
    x = fmaxf(x, __shfl_xor(x, offset, width));
  return x;
}

// ggml_cuda_mad: F32 += dot of two F16 pairs.
__device__ __forceinline__ void mad(float &acc, const __half2 v,
                                    const __half2 u) {
#if defined(HIP_TOKEN_MAJOR_GQA_TILE_DOT2)
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
  const float2 product = __half22float2(__hmul2(v, u));
  acc += product.x + product.y;
#endif
}

// n % divisor for the <multiplier, shift, divisor> triple of fastdivValues.
__device__ __forceinline__ uint32_t fastModulo(uint32_t n, const uint3 d) {
  const uint32_t quotient = (__umulhi(n, d.x) + n) >> d.y;
  return n - quotient * d.z;
}

//===----------------------------------------------------------------------===//
// Tile kernel (fattn-tile.cuh, FAST_FP16_AVAILABLE paths)
//===----------------------------------------------------------------------===//

// flash_attn_tile_load_tile: rows [0, I) of J halves from global memory into a
// shared tile with rows of J/2 + padding half2, zero past `rowLimit` when
// checked.
template <int nwarps, int I, int J, int padding, bool checked>
__device__ __forceinline__ void loadTile(const __half2 *__restrict__ source,
                                         __half2 *__restrict__ tile,
                                         int sourceStride, int rowLimit) {
  static_assert(J % 8 == 0 && (J / 2) % kCopyWords == 0, "bad J");
  auto load = [&](int n) {
    const int strideJ = kWarpSize >> n;
    if (strideJ == 0)
      return;
    const int j0Start =
        strideJ == kWarpSize
            ? 0
            : ((J / 2) / kCopyWords) - ((J / 2) / kCopyWords) % (2 * strideJ);
    const int j0Stop =
        ((J / 2) / kCopyWords) - ((J / 2) / kCopyWords) % (1 * strideJ);
    const int strideI = kWarpSize / strideJ;
    if (j0Start == j0Stop)
      return;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * strideI) {
      const int i = i0 + threadIdx.y * strideI +
                    (strideJ == kWarpSize ? 0 : threadIdx.x / strideJ);
      if (i0 + nwarps * strideI <= I || i < I) {
#pragma unroll
        for (int j0 = j0Start; j0 < j0Stop; j0 += strideJ) {
          const int j =
              j0 * kCopyWords +
              (strideJ == kWarpSize ? threadIdx.x : threadIdx.x % strideJ) *
                  kCopyWords;
          const __align__(16) __half2 zero[kCopyWords] = {};
          copyOnce<kCopyBytes>(
              tile + i * (J / 2 + padding) + j,
              !checked || i < rowLimit ? source + i * sourceStride + j : zero);
        }
      }
    }
  };
  Unroll<7>{}(load);
}

// flash_attn_tile_iter_KQ: accumulates nbatchK columns of K.Q for nbatchFa
// keys.
template <int nwarps, int ncols1, int ncols2, int DKQ, int nbatchFa,
          int nbatchK, bool checked>
__device__ __forceinline__ void
iterateKq(const __half2 *const queryTile, const __half2 *__restrict__ keys,
          __half2 *const kvTile, int keyStride, int key0, int keyLimit,
          int column0, float *kqAccumulator) {
  constexpr int ncols = ncols1 * ncols2;
  constexpr int cpw = ncols > nwarps ? ncols / nwarps : 1;
  constexpr int np = nwarps > ncols ? nwarps / ncols : 1;

  loadTile<nwarps, nbatchFa, nbatchK, kCopyWords, checked>(
      keys + int64_t(key0) * keyStride + column0 / 2, kvTile, keyStride,
      keyLimit);
  __syncthreads();

  static_assert((nbatchK / 2) % kCopyWords == 0, "bad nbatchK");
#pragma unroll
  for (int column1 = 0; column1 < nbatchK / 2; column1 += kCopyWords) {
    __align__(16) __half2 keyRegs[nbatchFa / (np * kWarpSize)][kCopyWords];
    __align__(16) __half2 queryRegs[cpw][kCopyWords];
#pragma unroll
    for (int i0 = 0; i0 < nbatchFa; i0 += np * kWarpSize) {
      const int i = i0 + (threadIdx.y % np) * kWarpSize + threadIdx.x;
      copyOnce<kCopyBytes>(&keyRegs[i0 / (np * kWarpSize)],
                           &kvTile[i * (nbatchK / 2 + kCopyWords) + column1]);
    }
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
      const int jc = jc0 + (threadIdx.y / np) * cpw;
      copyOnce<kCopyBytes>(&queryRegs[jc0],
                           &queryTile[jc * (DKQ / 2) + column0 / 2 + column1]);
    }
#pragma unroll
    for (int i0 = 0; i0 < nbatchFa; i0 += np * kWarpSize)
#pragma unroll
      for (int jc0 = 0; jc0 < cpw; ++jc0)
#pragma unroll
        for (int k = 0; k < kCopyWords; ++k)
          mad(kqAccumulator[i0 / (np * kWarpSize) * cpw + jc0],
              keyRegs[i0 / (np * kWarpSize)][k], queryRegs[jc0][k]);
  }

  if (column0 + nbatchK < DKQ)
    __syncthreads(); // Not needed after the last chunk.
}

// flash_attn_tile_iter: one pass over nbatchFa keys - scores, online softmax,
// and the V product.
template <int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatchFa,
          int nbatchK, bool checked>
__device__ __forceinline__ void
iterateKeys(const __half2 *const queryTile, const __half2 *__restrict__ keys,
            const __half2 *__restrict__ values, const __half *__restrict__ mask,
            const uint3 queryTokens, __half *const kqTile,
            __half2 *const kvTile, int keyStride, int valueStride,
            int maskStride, float *const kqMax, float *const kqSum,
            __half2 *const vkq, int key0, int keyEnd, int queryColumn0) {
  constexpr int ncols = ncols1 * ncols2;
  constexpr int cpw = ncols > nwarps ? ncols / nwarps : 1;
  constexpr int np = nwarps > ncols ? nwarps / ncols : 1;
  constexpr int DVp = (DV + 2 * kWarpSize - 1) & ~(2 * kWarpSize - 1);
  // Scores per query column stored as one contiguous chunk of kqTile, which
  // is laid out [ncols / kqChunk][nbatchFa][kqChunk].
  constexpr int kqChunk = cpw < 2 * kCopyWords ? cpw : 2 * kCopyWords;
  static_assert(cpw % kqChunk == 0, "bad kqChunk");
  const int keyLimit = keyEnd - key0;

  float kqMaxNew[cpw];
#pragma unroll
  for (int jc0 = 0; jc0 < cpw; ++jc0)
    kqMaxNew[jc0] = kqMax[jc0];

  float kqAccumulator[nbatchFa / (np * kWarpSize) * cpw] = {0.0f};

  constexpr int nbatchKLast = DKQ % nbatchK;
#pragma unroll
  for (int column0 = 0; column0 < DKQ - nbatchKLast; column0 += nbatchK)
    iterateKq<nwarps, ncols1, ncols2, DKQ, nbatchFa, nbatchK, checked>(
        queryTile, keys, kvTile, keyStride, key0, keyLimit, column0,
        kqAccumulator);
  if constexpr (nbatchKLast > 0)
    iterateKq<nwarps, ncols1, ncols2, DKQ, nbatchFa, nbatchKLast, checked>(
        queryTile, keys, kvTile, keyStride, key0, keyLimit, DKQ - nbatchKLast,
        kqAccumulator);

    // Mask, then the running maximum.
#pragma unroll
  for (int jc0 = 0; jc0 < cpw; ++jc0) {
    const int j = fastModulo(
        queryColumn0 + (jc0 + (threadIdx.y / np) * cpw) / ncols2, queryTokens);
#pragma unroll
    for (int i0 = 0; i0 < nbatchFa; i0 += np * kWarpSize) {
      const int i = i0 + (threadIdx.y % np) * kWarpSize + threadIdx.x;
      float &score = kqAccumulator[(i0 / (np * kWarpSize)) * cpw + jc0];
#if !defined(HIP_TOKEN_MAJOR_GQA_TILE_DOT2)
      // The query was scaled by 1/4 to keep the F16 products in range.
      score *= 4.0f;
#endif
      if (!checked || i < keyLimit) {
        score += (ncols2 > 1 || mask)
                     ? __half2float(mask[j * maskStride + key0 + i])
                     : 0.0f;
        kqMaxNew[jc0] = fmaxf(kqMaxNew[jc0], score + kKqMaxOffset);
      }
    }
    kqMaxNew[jc0] = warpReduceMax(kqMaxNew[jc0]);
  }

  if constexpr (np == 1) {
    __syncthreads();
  } else {
    static_assert(cpw == 1, "bad cpw");
    __shared__ float kqMaxShared[nwarps];
    if (threadIdx.x == 0)
      kqMaxShared[threadIdx.y] = kqMaxNew[0];
    __syncthreads();
    kqMaxNew[0] = kqMaxShared[(threadIdx.y & ~(np - 1)) + threadIdx.x % np];
    kqMaxNew[0] = warpReduceMax<np>(kqMaxNew[0]);
  }

  // Probabilities into kqTile; rescale the V accumulators.
#pragma unroll
  for (int jc0 = 0; jc0 < cpw; jc0 += kqChunk) {
    __align__(16) __half probabilities[nbatchFa / (np * kWarpSize)][kqChunk];
#pragma unroll
    for (int jc1 = 0; jc1 < kqChunk; ++jc1) {
      const int jc = jc0 + jc1;
      const float rescale = expf(kqMax[jc] - kqMaxNew[jc]);
      kqMax[jc] = kqMaxNew[jc];
      float sumAdd = 0.0f;
#pragma unroll
      for (int i0 = 0; i0 < nbatchFa; i0 += np * kWarpSize) {
        const float value =
            !checked || i0 + (threadIdx.y % np) * kWarpSize + threadIdx.x <
                            static_cast<uint32_t>(keyLimit)
                ? expf(kqAccumulator[(i0 / (np * kWarpSize)) * cpw + jc] -
                       kqMax[jc])
                : 0.0f;
        sumAdd += value;
        probabilities[i0 / (np * kWarpSize)][jc1] = __float2half(value);
      }
      kqSum[jc] = kqSum[jc] * rescale + sumAdd;
      const __half2 rescale2 = __float2half2_rn(rescale);
#pragma unroll
      for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize)
        vkq[jc * ((DVp / 2) / kWarpSize) + i0 / kWarpSize] = __hmul2(
            vkq[jc * ((DVp / 2) / kWarpSize) + i0 / kWarpSize], rescale2);
    }
#pragma unroll
    for (int i0 = 0; i0 < nbatchFa; i0 += np * kWarpSize) {
      const int i = i0 + (threadIdx.y % np) * kWarpSize + threadIdx.x;
      copyOnce<sizeof(probabilities[0])>(
          kqTile +
              (jc0 / kqChunk + (threadIdx.y / np) * (cpw / kqChunk)) *
                  (nbatchFa * kqChunk) +
              i * kqChunk,
          probabilities[i0 / (np * kWarpSize)]);
    }
  }

  // VKQ += V . probabilities
  static_assert(DV <= DKQ, "bad DV");
  static_assert(DV % nbatchK == 0 ||
                    (nbatchK % 3 == 0 && DV % (nbatchK * 2 / 3) == 0),
                "bad nbatchK");
  // V rows that fit in the shared tile sized for K.
  constexpr int nbatchV =
      (DV % nbatchK == 0 ? nbatchK : nbatchK * 2 / 3) * nbatchFa / DV;
  static_assert(nbatchFa % nbatchV == 0 && nbatchV % np == 0, "bad nbatchV");
#pragma unroll
  for (int k0 = 0; k0 < nbatchFa; k0 += nbatchV) {
    loadTile<nwarps, nbatchV, DV, 0, checked>(
        values + int64_t(key0 + k0) * valueStride, kvTile, valueStride,
        keyLimit - k0);
    __syncthreads();

#pragma unroll
    for (int k1 = 0; k1 < nbatchV; k1 += np) {
      __align__(16) __half2 valueRegs[(DVp / 2) / kWarpSize];
      __align__(16) __half2 probabilityRegs[cpw];

      constexpr int copyD = kCopyWords / 2 < (DVp / 2) / kWarpSize
                                ? kCopyWords / 2
                                : (DVp / 2) / kWarpSize;
#pragma unroll
      for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize * copyD)
        copyOnce<copyD * 4>(&valueRegs[i0 / kWarpSize],
                            &kvTile[(k1 + threadIdx.y % np) * (DV / 2) + i0 +
                                    threadIdx.x * copyD]);
#pragma unroll
      for (int jc0 = 0; jc0 < cpw; jc0 += kqChunk) {
        const int chunk = jc0 / kqChunk + (threadIdx.y / np) * (cpw / kqChunk);
        __align__(16) __half chunkValues[kqChunk];
        copyOnce<kqChunk * sizeof(__half)>(
            &chunkValues, kqTile + chunk * (nbatchFa * kqChunk) +
                              (k0 + k1 + threadIdx.y % np) * kqChunk);
#pragma unroll
        for (int jc1 = 0; jc1 < kqChunk; ++jc1)
          probabilityRegs[jc0 + jc1] = __half2half2(chunkValues[jc1]);
      }
#pragma unroll
      for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize)
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0)
          vkq[jc0 * ((DVp / 2) / kWarpSize) + i0 / kWarpSize] =
              __hfma2(valueRegs[i0 / kWarpSize], probabilityRegs[jc0],
                      vkq[jc0 * ((DVp / 2) / kWarpSize) + i0 / kWarpSize]);
    }
    __syncthreads();
  }
}

// Where a block finds the live key extent of its (sequence, query tile):
// tileKvExtentKernel's output when it ran, or else the rows the cache update
// wrote, which bound what any query can see, the same way that kernel derives
// them. Reading those rows in the block saves a launch per attention call.
struct KeyExtent {
  const int *kvMax;
  const int64_t *writtenRows;
  int32_t writtenRowCount;
  int32_t liveKeys;
  int32_t cacheRows;
  int32_t granularity;
};

// Each warp reads the rows itself. A shared maximum (one more shared
// variable and two barriers, even when the kvMax path is taken) made the
// compiled tile kernel's key loop about 60% slower over 64K keys.
__device__ int blockKeyExtent(const KeyExtent &extent, int index) {
  if (extent.kvMax)
    return extent.kvMax[index];
  if (!extent.writtenRows)
    return extent.cacheRows;
  int highest = 0;
  for (int i = threadIdx.x % kWarpSize; i < extent.writtenRowCount;
       i += kWarpSize) {
    const int64_t row = extent.writtenRows[i];
    if (row >= 0 && row + 1 > highest)
      highest = static_cast<int>(row + 1);
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    highest = max(highest, __shfl_xor(highest, offset, kWarpSize));
  const int bound = min(extent.liveKeys, highest);
  const int rounded = (bound + extent.granularity - 1) / extent.granularity *
                      extent.granularity;
  return min(rounded, extent.cacheRows);
}

// flash_attn_tile. Block (blockIdx.x, blockIdx.y, blockIdx.z) handles query
// tokens [x * ncols1, +ncols1) for query heads [h0, h0 + ncols2) of one
// sequence, over key tiles y, y + gridDim.y, ... up to kvMax. With
// gridDim.y > 1 it writes unnormalized partial results and their (max, sum)
// to `partial` / `partialMeta`, merged by tileCombineKernel.
//
// The heads a block groups share one KV head. A group whose size ncols2 does
// not divide (three query heads a KV head: ncols2 4) is padded: z counts the
// KV heads' groups, and a column past the KV head's last query head reads a
// zero query and writes nothing - the KV head's keys and values are read
// once, not once a query head.
template <int DKQ, int DV, int ncols1, int ncols2>
__launch_bounds__(tileConfig(kDeviceRdna, DKQ, ncols1 *ncols2).nthreads,
                  tileConfig(kDeviceRdna, DKQ, ncols1 *ncols2).occupancy)
    __global__ void flashAttentionTileKernel(
        const char *__restrict__ queryBytes, const char *__restrict__ keyBytes,
        const char *__restrict__ valueBytes, const char *__restrict__ maskBytes,
        const KeyExtent keyExtent, float *__restrict__ partial,
        float2 *__restrict__ partialMeta, const float scale,
        const uint3 queryTokens, const int32_t queryHeads,
        const int32_t queryTokenStride, const int32_t queryHeadStride,
        const int32_t querySequenceStride, const int32_t cacheRows,
        const int32_t kvHeads, const int32_t keyRowStride,
        const int32_t keyHeadStride, const int64_t keySequenceStride,
        const int32_t valueRowStride, const int32_t valueHeadStride,
        const int64_t valueSequenceStride, const int32_t maskBatches,
        const int32_t maskRowStride, const int64_t maskBatchStride) {
  constexpr TileConfig config = tileConfig(kDeviceRdna, DKQ, ncols1 * ncols2);
  static_assert(config.nthreads != 0, "kernel config not defined");
  constexpr int ncols = ncols1 * ncols2;
  constexpr int nwarps = config.nthreads / kWarpSize;
  constexpr int nbatchFa = config.nbatchFa;
  constexpr int nbatchK = config.nbatchK;

  const int queryColumn0 = blockIdx.x * ncols1;
  const int gqaRatio = queryHeads / kvHeads;
  const int groupsPerKvHead = (gqaRatio + ncols2 - 1) / ncols2;
  const int sequence = blockIdx.z / (kvHeads * groupsPerKvHead);
  const int kvHead = blockIdx.z / groupsPerKvHead - sequence * kvHeads;
  const int group = blockIdx.z % groupsPerKvHead;
  const int head0 = kvHead * gqaRatio + group * ncols2;
  // The block's query heads: fewer than ncols2 in a padded group.
  const int heads = min(ncols2, gqaRatio - group * ncols2);
  const float *query = reinterpret_cast<const float *>(
      queryBytes + int64_t(querySequenceStride) * sequence +
      int64_t(queryHeadStride) * head0);
  const __half2 *keys = reinterpret_cast<const __half2 *>(
      keyBytes + keySequenceStride * sequence +
      int64_t(keyHeadStride) * kvHead);
  const __half2 *values = reinterpret_cast<const __half2 *>(
      valueBytes + valueSequenceStride * sequence +
      int64_t(valueHeadStride) * kvHead);
  const __half *mask =
      maskBytes ? reinterpret_cast<const __half *>(
                      maskBytes + maskBatchStride * (sequence % maskBatches))
                : nullptr;
  const int keyStride = keyRowStride / static_cast<int>(sizeof(__half2));
  const int valueStride = valueRowStride / static_cast<int>(sizeof(__half2));
  const int maskStride = maskRowStride / static_cast<int>(sizeof(__half));

  constexpr int cpw = ncols > nwarps ? ncols / nwarps : 1; // columns per warp
  constexpr int np = nwarps > ncols ? nwarps / ncols : 1;  // warps per column
  static_assert(cpw == 1 || np == 1, "bad cpw / np");
  static_assert(nbatchFa % (np * kWarpSize) == 0, "bad nbatchFa");
  constexpr int DKQp = (DKQ + 2 * kWarpSize - 1) & ~(2 * kWarpSize - 1);
  constexpr int DVp = (DV + 2 * kWarpSize - 1) & ~(2 * kWarpSize - 1);

  // queryTile: the block's queries for the whole kernel. kvTile: K or V
  // rows, padded against bank conflicts (K) and out-of-range reads (V).
  // kqTile: probabilities between the two products. vkq: accumulators.
  __shared__ __half2 queryTile[ncols * DKQ / 2];
  __shared__ __half2 kvTile[nbatchFa * (nbatchK / 2 + kCopyWords) + DVp - DV];
  __shared__ __half kqTile[ncols * nbatchFa];
  __align__(16) __half2 vkq[cpw * ((DVp / 2) / kWarpSize)] = {};

  float kqMax[cpw];
#pragma unroll
  for (int j0 = 0; j0 < ncols; j0 += nwarps)
    kqMax[j0 / nwarps] = -FLT_MAX / 2.0f;
  float kqSum[cpw] = {0.0f};

  // Queries, scaled, to F16.
#pragma unroll
  for (int jc0 = 0; jc0 < cpw; ++jc0) {
    const int jc = jc0 + (threadIdx.y / np) * cpw;
    const int j = jc / ncols2;
    const int c = jc % ncols2;
    constexpr int copyD =
        kCopyWords < DKQp / kWarpSize ? kCopyWords : DKQp / kWarpSize;
#pragma unroll
    for (int i0 = 0; i0 < DKQp; i0 += np * kWarpSize * copyD) {
      const int offset =
          i0 + (threadIdx.y % np) * (kWarpSize * copyD) + threadIdx.x * copyD;
      if (i0 + np * kWarpSize * copyD <= DKQ || offset < DKQ) {
        __align__(16) float scaled[copyD] = {0.0f};
        if (c < heads)
          copyOnce<sizeof(scaled)>(
              scaled,
              &query[c * (queryHeadStride / static_cast<int>(sizeof(float))) +
                     fastModulo(queryColumn0 + j, queryTokens) *
                         (queryTokenStride / static_cast<int>(sizeof(float))) +
                     offset]);
#pragma unroll
        for (int i1 = 0; i1 < copyD; ++i1)
          scaled[i1] *= scale;
        __align__(16) __half2 halves[copyD / 2];
#pragma unroll
        for (int i1 = 0; i1 < copyD; i1 += 2) {
          halves[i1 / 2] = __floats2half2_rn(scaled[i1], scaled[i1 + 1]);
#if !defined(HIP_TOKEN_MAJOR_GQA_TILE_DOT2)
          // Without v_dot2_f32_f16 the F16 products could overflow; the
          // scores are scaled back by 4 after the product.
          halves[i1 / 2] = __hmul2(halves[i1 / 2], __float2half2_rn(0.25f));
#endif
        }
        copyOnce<sizeof(halves)>(&queryTile[jc * (DKQ / 2) + offset / 2],
                                 halves);
      }
    }
  }
  __syncthreads();

  // Main loop over the live keys.
  const int keyEnd =
      blockKeyExtent(keyExtent, sequence * gridDim.x + blockIdx.x);
  if constexpr (ncols2 == 1) {
    // Ungrouped heads check every key against keyEnd.
    int key0 = blockIdx.y * nbatchFa;
    while (key0 < keyEnd - nbatchFa) {
      iterateKeys<nwarps, ncols1, ncols2, DKQ, DV, nbatchFa, nbatchK, false>(
          queryTile, keys, values, mask, queryTokens, kqTile, kvTile, keyStride,
          valueStride, maskStride, kqMax, kqSum, vkq, key0, keyEnd,
          queryColumn0);
      key0 += gridDim.y * nbatchFa;
    }
    if (key0 < keyEnd)
      iterateKeys<nwarps, ncols1, ncols2, DKQ, DV, nbatchFa, nbatchK, true>(
          queryTile, keys, values, mask, queryTokens, kqTile, kvTile, keyStride,
          valueStride, maskStride, kqMax, kqSum, vkq, key0, keyEnd,
          queryColumn0);
  } else if constexpr (DKQ == DV) {
    // Grouped heads read whole tiles; the mask rejects keys past the extent,
    // which is whole tiles: the cache is whole kKqStride tiles (visitShape)
    // and the extent is rounded to nbatchFa. Without a checked tail the
    // kernel keeps llama.cpp's registers - one costs it two waves a SIMD.
    for (int key0 = blockIdx.y * nbatchFa; key0 < keyEnd;
         key0 += gridDim.y * nbatchFa)
      iterateKeys<nwarps, ncols1, ncols2, DKQ, DV, nbatchFa, nbatchK, false>(
          queryTile, keys, values, mask, queryTokens, kqTile, kvTile, keyStride,
          valueStride, maskStride, kqMax, kqSum, vkq, key0, keyEnd,
          queryColumn0);
  } else {
    // The compressed-key shape groups heads over a cache of any capacity
    // (visitCompressedShape) and checks the last tile of one that is not
    // whole tiles.
    int key0 = blockIdx.y * nbatchFa;
    while (key0 + nbatchFa <= keyEnd) {
      iterateKeys<nwarps, ncols1, ncols2, DKQ, DV, nbatchFa, nbatchK, false>(
          queryTile, keys, values, mask, queryTokens, kqTile, kvTile, keyStride,
          valueStride, maskStride, kqMax, kqSum, vkq, key0, keyEnd,
          queryColumn0);
      key0 += gridDim.y * nbatchFa;
    }
    if (key0 < keyEnd)
      iterateKeys<nwarps, ncols1, ncols2, DKQ, DV, nbatchFa, nbatchK, true>(
          queryTile, keys, values, mask, queryTokens, kqTile, kvTile, keyStride,
          valueStride, maskStride, kqMax, kqSum, vkq, key0, keyEnd,
          queryColumn0);
  }

#pragma unroll
  for (int jc0 = 0; jc0 < cpw; ++jc0)
    kqSum[jc0] = warpReduceSum(kqSum[jc0]);

  if constexpr (np > 1) {
    // Warps sharing a column add their accumulators into the first one.
    static_assert(cpw == 1, "bad cpw");
    static_assert(nbatchFa * nbatchK >= nwarps * DVp, "kvTile too small");
    __half2 *vkqCombine = kvTile;
    float *kqSumCombine = reinterpret_cast<float *>(queryTile);
    constexpr int copyD =
        kCopyWords < (DVp / 2) / kWarpSize ? kCopyWords : (DVp / 2) / kWarpSize;
    if (threadIdx.y % np != 0) {
#pragma unroll
      for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize * copyD)
        copyOnce<copyD * 4>(
            &vkqCombine[threadIdx.y * (DVp / 2) + i0 + threadIdx.x * copyD],
            &vkq[i0 / kWarpSize]);
      if (threadIdx.x == 0)
        kqSumCombine[threadIdx.y] = kqSum[0];
      return;
    }
    __syncthreads();
#pragma unroll
    for (int ip = 1; ip < np; ++ip) {
#pragma unroll
      for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize * copyD) {
        __align__(16) __half2 other[copyD];
        copyOnce<copyD * 4>(other, &vkqCombine[(threadIdx.y + ip) * (DVp / 2) +
                                               i0 + threadIdx.x * copyD]);
#pragma unroll
        for (int i1 = 0; i1 < copyD; ++i1)
          vkq[i0 / kWarpSize + i1] =
              __hadd2(vkq[i0 / kWarpSize + i1], other[i1]);
      }
      kqSum[0] += kqSumCombine[threadIdx.y + ip];
    }
  }

  // Results: normalized when there is one parallel block, else partials.
#pragma unroll
  for (int jc0 = 0; jc0 < cpw; ++jc0) {
    const int jc = jc0 + (threadIdx.y / np) * cpw;
    const int j = jc / ncols2;
    const int c = jc % ncols2;
    if (ncols1 > 1 && queryColumn0 + j >= int(queryTokens.z))
      return;
    if (c >= heads)
      continue;
    const float normalize = gridDim.y == 1 ? 1.0f / kqSum[jc0] : 1.0f;
    const int row =
        ((sequence * int(queryTokens.z) + queryColumn0 + j) * queryHeads +
         head0 + c) *
            gridDim.y +
        blockIdx.y;
    constexpr int copyD = kCopyWords / 2 < (DVp / 2) / kWarpSize
                              ? kCopyWords / 2
                              : (DVp / 2) / kWarpSize;
#pragma unroll
    for (int i0 = 0; i0 < DVp / 2; i0 += kWarpSize * copyD) {
      __align__(16) float2 out[copyD];
#pragma unroll
      for (int i1 = 0; i1 < copyD; ++i1) {
        out[i1] = __half22float2(
            vkq[jc0 * ((DVp / 2) / kWarpSize) + i0 / kWarpSize + i1]);
        out[i1].x *= normalize;
        out[i1].y *= normalize;
      }
      if (i0 + kWarpSize * copyD <= DV / 2 || i0 + threadIdx.x * copyD < DV / 2)
        copyOnce<sizeof(out)>(
            &partial[row * DV + 2 * i0 + threadIdx.x * (2 * copyD)], out);
    }
    if (gridDim.y != 1 && threadIdx.x == 0)
      partialMeta[row] = make_float2(kqMax[jc0], kqSum[jc0]);
  }
}

// The live key extent of each (sequence, query tile): one past the highest
// row any query of the tile can see, rounded up to `granularity`. Block z
// scans keys [z * kExtentKeysPerBlock, +kExtentKeysPerBlock) and raises the
// zeroed kvMax to what it saw, so a capacity far past the live keys costs
// blocks rather than a serial walk down to them.
__global__ void __launch_bounds__(kExtentThreads)
    tileKvExtentKernel(const __half *__restrict__ mask, int *__restrict__ kvMax,
                       int64_t liveKeys, int64_t cacheCapacity,
                       int64_t queryTokens, int ncols1, int64_t maskBatchCount,
                       int granularity) {
  const int tile = blockIdx.x;
  const int sequence = blockIdx.y;
  const int64_t first = static_cast<int64_t>(blockIdx.z) * kExtentKeysPerBlock;
  const int64_t end = std::min<int64_t>(liveKeys, first + kExtentKeysPerBlock);
  const __half *maskBatch =
      mask ? mask + (sequence % maskBatchCount) * queryTokens * cacheCapacity
           : nullptr;
  const int64_t firstRow = static_cast<int64_t>(tile) * ncols1;
  const int64_t endRow = std::min<int64_t>(queryTokens, firstRow + ncols1);
  int highest = 0;
  for (int64_t key = first + threadIdx.x; key < end; key += kExtentThreads) {
    bool visible = !maskBatch;
    for (int64_t row = firstRow; row < endRow && !visible; ++row) {
      const float value = __half2float(maskBatch[row * cacheCapacity + key]);
      visible = !(isinf(value) && value < 0.0f);
    }
    if (visible)
      highest = static_cast<int>(key + 1);
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    highest = max(highest, __shfl_xor(highest, offset, kWarpSize));
  if (highest > 0 && threadIdx.x % kWarpSize == 0) {
    const int64_t rounded =
        (highest + granularity - 1) / granularity * granularity;
    atomicMax(&kvMax[sequence * gridDim.x + tile],
              static_cast<int>(std::min<int64_t>(rounded, cacheCapacity)));
  }
}

// llama.cpp's flash_attn_combine_results, except that it reads only the
// parallel blocks that had keys to process (block y covers key tiles y,
// y + parallelBlocks, ...), a row with no visible key (every partial sum
// zero) is written as zeros instead of 0/0.
template <int D>
__global__ void __launch_bounds__(D)
    tileCombineKernel(const float *__restrict__ parts,
                      const float2 *__restrict__ meta, float *__restrict__ dst,
                      const KeyExtent keyExtent, int parallelBlocks, int ncols1,
                      int nbatchFa) {
  // Grid (H, T, B); rows are [B][T][H] like the output.
  const int64_t row =
      (static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x +
      blockIdx.x;
  const int tilesX = (gridDim.y + ncols1 - 1) / ncols1;
  const int extent =
      blockKeyExtent(keyExtent, blockIdx.z * tilesX + blockIdx.y / ncols1);
  const int active =
      max(1, min(parallelBlocks, (extent + nbatchFa - 1) / nbatchFa));
  parts += row * parallelBlocks * D;
  meta += row * parallelBlocks;
  dst += row * D;

  __shared__ float2 sharedMeta[kMaximumParallelBlocks];
  for (int l = threadIdx.x; l < active; l += D)
    sharedMeta[l] = meta[l];
  __syncthreads();

  float maximum = -FLT_MAX;
  for (int l = 0; l < active; ++l)
    maximum = fmaxf(maximum, sharedMeta[l].x);
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (int l = 0; l < active; ++l) {
    const float weight = expf(sharedMeta[l].x - maximum);
    numerator += weight * parts[l * D + threadIdx.x];
    denominator += weight * sharedMeta[l].y;
  }
  dst[threadIdx.x] = denominator > 0.0f ? numerator / denominator : 0.0f;
}

// Mirrors GGML_CUDA_CC_IS_RDNA for the architectures whose device code defines
// RDNA, so the host reads the same configuration table the kernel was built
// with.
bool deviceIsRdna(const char *arch) {
  if (std::strncmp(arch, "gfx11", 5) == 0 ||
      std::strncmp(arch, "gfx12", 5) == 0)
    return true;
  if (std::strncmp(arch, "gfx1010", 7) == 0 ||
      std::strncmp(arch, "gfx1012", 7) == 0)
    return true;
  return std::strncmp(arch, "gfx103", 6) == 0 && arch[6] >= '0' &&
         arch[6] <= '7';
}

struct DeviceInfo {
  bool valid = false;
  bool rdna = false;
  int multiprocessors = 0;
};

DeviceInfo currentDevice() {
  constexpr int kCachedDevices = 16;
  static DeviceInfo cache[kCachedDevices];
  static std::atomic<bool> ready[kCachedDevices]{};
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess || device < 0)
    return {};
  if (device < kCachedDevices && ready[device].load(std::memory_order_acquire))
    return cache[device];
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess ||
      properties.warpSize != kWarpSize)
    return {};
  DeviceInfo info;
  info.valid = true;
  info.rdna = deviceIsRdna(properties.gcnArchName);
  info.multiprocessors = properties.multiProcessorCount;
  if (device < kCachedDevices) {
    cache[device] = info;
    ready[device].store(true, std::memory_order_release);
  }
  return info;
}

// init_fastdiv_values without its GGML_ASSERT, which would tie this library
// to ggml's abort handler; callers pass a validated non-zero divisor.
uint3 fastdivValues(uint32_t d) {
  uint32_t L = 0;
  while (L < 32 && (uint32_t{1} << L) < d)
    ++L;
  const uint32_t mp = static_cast<uint32_t>(
      (uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
  return make_uint3(mp, L, d);
}

template <int DKQ_, int DV_, int ncols1_, int ncols2_> struct TileShape {
  static constexpr int DKQ = DKQ_;
  static constexpr int DV = DV_;
  static constexpr int ncols1 = ncols1_;
  static constexpr int ncols2 = ncols2_;
};

// What a launch of one tile instantiation looks like for a call: grid, the
// parallel-block count llama.cpp's launch_fattn would choose for the
// capacity, and the scratch that count needs.
struct TilePlan {
  int nthreads = 0;
  int nbatchFa = 0;
  int parallelBlocks = 0;
  int64_t tilesX = 0;
  int64_t tilesZ = 0;
  int64_t rows = 0;
  size_t extentBytes = 0;
  size_t bytesPerBlock = 0;
  size_t scratchBytes() const {
    return extentBytes + static_cast<size_t>(parallelBlocks) * bytesPerBlock;
  }
};

template <typename Shape>
int planTile(const TileAttentionArgs &a, const DeviceInfo &device,
             TilePlan &plan) {
  constexpr int DKQ = Shape::DKQ, DV = Shape::DV, ncols1 = Shape::ncols1,
                ncols2 = Shape::ncols2;
  const TileConfig config = tileConfig(device.rdna, DKQ, ncols1 * ncols2);
  if (config.nthreads == 0)
    return kTileUnsupported;
  plan.nthreads = config.nthreads;
  plan.nbatchFa = config.nbatchFa;

  auto *kernel = flashAttentionTileKernel<DKQ, DV, ncols1, ncols2>;
  int blocksPerMultiprocessor = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocksPerMultiprocessor, reinterpret_cast<const void *>(kernel),
          plan.nthreads, 0) != hipSuccess ||
      blocksPerMultiprocessor <= 0) {
    (void)hipGetLastError();
    return kTileUnsupported;
  }

  const int64_t gqaRatio = a.queryHeads / a.kvHeads;
  plan.tilesX = (a.queryTokens + ncols1 - 1) / ncols1;
  plan.tilesZ = (gqaRatio + ncols2 - 1) / ncols2 * a.kvHeads * a.batchCount;
  const int64_t tilesDst = plan.tilesX * plan.tilesZ;
  const int64_t tilesKv = (a.cacheCapacity + plan.nbatchFa - 1) / plan.nbatchFa;

  // llama.cpp's parallel-block choice, made from the capacity. At least two
  // keep the combine kernel, and with it the all-masked case, on every path.
  int parallelBlocks =
      static_cast<int>(std::min<int64_t>(blocksPerMultiprocessor, tilesKv));
  const int64_t blocksPerWave =
      static_cast<int64_t>(device.multiprocessors) * blocksPerMultiprocessor;
  int bestWaves = 0;
  int bestEfficiency = 0;
  for (int64_t candidate = parallelBlocks; candidate <= tilesKv; ++candidate) {
    const int64_t total = tilesDst * candidate;
    const int64_t waves = (total + blocksPerWave - 1) / blocksPerWave;
    const int efficiency =
        static_cast<int>(100 * total / (waves * blocksPerWave));
    if (bestEfficiency >= 95 && waves > bestWaves)
      break;
    if (efficiency > bestEfficiency) {
      bestWaves = static_cast<int>(waves);
      bestEfficiency = efficiency;
      parallelBlocks = static_cast<int>(candidate);
    }
  }
  plan.parallelBlocks =
      std::max(2, std::min(parallelBlocks, kMaximumParallelBlocks));

  plan.rows = a.batchCount * a.queryTokens * a.queryHeads;
  plan.extentBytes =
      static_cast<size_t>(plan.tilesX * a.batchCount) * sizeof(int);
  plan.bytesPerBlock =
      static_cast<size_t>(plan.rows) * (DV * sizeof(float) + sizeof(float2));

  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t nb01 = a.queryHeads * DKQ * sizeof(float);
  const int64_t cacheBytes =
      a.cacheCapacity * a.kvHeads * DKQ * sizeof(__half);
  if (a.cacheCapacity > kInt32Max || a.queryTokens > kInt32Max ||
      a.writtenRowCount > kInt32Max || a.queryTokens * nb01 > kInt32Max ||
      cacheBytes > kInt32Max ||
      a.cacheCapacity * a.queryTokens * sizeof(__half) > kInt32Max ||
      plan.tilesX > std::numeric_limits<unsigned>::max() || plan.tilesZ > 65535)
    return kTileUnsupported;
  return 0;
}

template <typename Shape>
int launchTile(const TileAttentionArgs &a, const DeviceInfo &device) {
  constexpr int DKQ = Shape::DKQ, DV = Shape::DV, ncols1 = Shape::ncols1,
                ncols2 = Shape::ncols2;
  TilePlan plan;
  if (int status = planTile<Shape>(a, device, plan))
    return status;
  // Partial results, their (max, sum) pairs and the extents share the
  // caller's scratch, which may hold fewer parallel blocks than planned.
  if (!a.scratch || a.scratchBytes < plan.extentBytes + 2 * plan.bytesPerBlock)
    return kTileUnsupported;
  const int parallelBlocks = static_cast<int>(
      std::min<size_t>((a.scratchBytes - plan.extentBytes) / plan.bytesPerBlock,
                       static_cast<size_t>(plan.parallelBlocks)));

  auto *parts = static_cast<float *>(a.scratch);
  auto *meta =
      reinterpret_cast<float2 *>(parts + plan.rows * parallelBlocks * DV);
  auto *kvMax = reinterpret_cast<int *>(meta + plan.rows * parallelBlocks);

  // Byte strides of a token-major cache's row, KV head, and sequence.
  struct Strides {
    int64_t row, head, sequence;
  };
  const int64_t width = DKQ * sizeof(__half);
  const Strides key{a.kvHeads * width, width,
                    a.cacheCapacity * a.kvHeads * width};
  const Strides value = key;
  const int64_t queryTokenStride = a.queryHeads * DKQ * sizeof(float);
  const int64_t maskRowStride = a.cacheCapacity * sizeof(__half);

  // Without the per-key bounds check (ncols2 > 1) the kernel reads whole
  // nbatchFa tiles, so the extent is rounded to them when the capacity is
  // whole tiles; otherwise (compressed keys) it checks the last.
  const int granularity =
      ncols2 > 1 && a.cacheCapacity % plan.nbatchFa == 0 ? plan.nbatchFa : 1;

  // With written rows each block finds its extent itself; only a mask scan
  // needs the extent kernel.
  KeyExtent keyExtent{
      nullptr,
      a.writtenRows,
      static_cast<int32_t>(a.writtenRowCount),
      static_cast<int32_t>(std::min<int64_t>(a.liveKeys, a.cacheCapacity)),
      static_cast<int32_t>(a.cacheCapacity),
      granularity};
  (void)hipGetLastError();
  if (!a.writtenRows) {
    keyExtent.kvMax = kvMax;
    const int64_t liveKeys = std::min<int64_t>(a.liveKeys, a.cacheCapacity);
    if (hipError_t status =
            hipMemsetAsync(kvMax, 0, plan.extentBytes, a.stream))
      return static_cast<int>(status);
    hipLaunchKernelGGL(tileKvExtentKernel,
                       dim3(static_cast<unsigned>(plan.tilesX),
                            static_cast<unsigned>(a.batchCount),
                            static_cast<unsigned>(std::max<int64_t>(
                                1, (liveKeys + kExtentKeysPerBlock - 1) /
                                       kExtentKeysPerBlock))),
                       dim3(kExtentThreads), 0, a.stream, a.mask, kvMax,
                       liveKeys, a.cacheCapacity, a.queryTokens, ncols1,
                       std::max<int64_t>(a.maskBatchCount, 1), granularity);
  }
  hipLaunchKernelGGL(
      (flashAttentionTileKernel<DKQ, DV, ncols1, ncols2>),
      dim3(static_cast<unsigned>(plan.tilesX),
           static_cast<unsigned>(parallelBlocks),
           static_cast<unsigned>(plan.tilesZ)),
      dim3(kWarpSize, plan.nthreads / kWarpSize), 0, a.stream,
      reinterpret_cast<const char *>(a.query),
      reinterpret_cast<const char *>(a.keyCache),
      reinterpret_cast<const char *>(a.valueCache),
      reinterpret_cast<const char *>(a.mask), keyExtent, parts, meta, a.scale,
      fastdivValues(static_cast<uint32_t>(a.queryTokens)),
      static_cast<int32_t>(a.queryHeads),
      static_cast<int32_t>(queryTokenStride),
      static_cast<int32_t>(DKQ * sizeof(float)),
      static_cast<int32_t>(a.queryTokens * queryTokenStride),
      static_cast<int32_t>(a.cacheCapacity), static_cast<int32_t>(a.kvHeads),
      static_cast<int32_t>(key.row), static_cast<int32_t>(key.head),
      key.sequence, static_cast<int32_t>(value.row),
      static_cast<int32_t>(value.head), value.sequence,
      static_cast<int32_t>(std::max<int64_t>(a.maskBatchCount, 1)),
      static_cast<int32_t>(maskRowStride), a.queryTokens * maskRowStride);
  hipLaunchKernelGGL(tileCombineKernel<DV>,
                     dim3(static_cast<unsigned>(a.queryHeads),
                          static_cast<unsigned>(a.queryTokens),
                          static_cast<unsigned>(a.batchCount)),
                     dim3(DV), 0, a.stream, parts, meta, a.output,
                     keyExtent, parallelBlocks, ncols1, plan.nbatchFa);
  return static_cast<int>(hipGetLastError());
}

// The query tile llama.cpp's launcher picks (launch_fattn_tile_switch_ncols1
// on HIP): ncols2 query heads of one KV group, and as many query tokens as
// the narrowest of 64 (D <= 128), 32, 16, 8, 4 or 2 columns that holds more
// than half the step's - a step of 4 tokens reads its keys in tiles of 8
// columns, not of the widest.
template <int D, int ncols2, typename Visitor>
int visitColumns(const TileAttentionArgs &a, Visitor &&visit) {
  const int64_t tokens = a.queryTokens;
  if constexpr (D <= 128) {
    if (tokens > 32 / ncols2)
      return visit(TileShape<D, D, 64 / ncols2, ncols2>{});
  }
  if (tokens > 16 / ncols2)
    return visit(TileShape<D, D, 32 / ncols2, ncols2>{});
  if constexpr (ncols2 <= 8) {
    if (tokens > 8 / ncols2)
      return visit(TileShape<D, D, 16 / ncols2, ncols2>{});
    if (tokens > 4 / ncols2)
      return visit(TileShape<D, D, 8 / ncols2, ncols2>{});
  }
  if constexpr (ncols2 <= 4) {
    if (tokens > 2 / ncols2)
      return visit(TileShape<D, D, 4 / ncols2, ncols2>{});
  }
  if constexpr (ncols2 <= 2)
    return visit(TileShape<D, D, 2 / ncols2, ncols2>{});
  else
    return visit(TileShape<D, D, 1, ncols2>{});
}

template <int D, typename Visitor>
int visitShape(const TileAttentionArgs &a, Visitor &&visit) {
  const int64_t ratio = a.queryHeads / a.kvHeads;
  // Grouping heads skips the per-key bounds check, which needs a mask and
  // whole kKqStride tiles (llama.cpp's use_gqa_opt).
  const bool grouped = a.mask && a.cacheCapacity % kKqStride == 0;
  if (grouped && ratio % 8 == 0)
    return visitColumns<D, 8>(a, visit);
  if (grouped && ratio % 4 == 0)
    return visitColumns<D, 4>(a, visit);
  if (grouped && ratio % 2 == 0)
    return visitColumns<D, 2>(a, visit);
  // A group no power of two divides (three query heads a KV head) is padded
  // to the next one when a step has few tokens: decoding reads the cache once
  // a KV head instead of once a query head, at the cost of idle columns a
  // longer step, bounded by its arithmetic, would pay for.
  if (grouped && ratio > 1 && ratio < 8 && a.queryTokens <= kPaddedGroupTokens)
    return ratio <= 4 ? visitColumns<D, 4>(a, visit)
                      : visitColumns<D, 8>(a, visit);
  if constexpr (D <= 256)
    return visitColumns<D, 1>(a, visit);
  return kTileUnsupported;
}

// Keep in step with hip_token_major_gqa_head_dimension_valid.
template <typename Visitor>
int visitTile(const TileAttentionArgs &a, Visitor &&visit) {
  switch (a.headDimension) {
  case 40:
    return visitShape<40>(a, visit);
  case 64:
    return visitShape<64>(a, visit);
  case 72:
    return visitShape<72>(a, visit);
  case 80:
    return visitShape<80>(a, visit);
  case 96:
    return visitShape<96>(a, visit);
  case 112:
    return visitShape<112>(a, visit);
  case 128:
    return visitShape<128>(a, visit);
  case 256:
    return visitShape<256>(a, visit);
  default:
    return kTileUnsupported;
  }
}

} // namespace

int launchTileAttention(const TileAttentionArgs &a) {
  const DeviceInfo device = currentDevice();
  if (!device.valid)
    return kTileUnsupported;
  return visitTile(
      a, [&](auto shape) { return launchTile<decltype(shape)>(a, device); });
}

size_t tileAttentionScratchBytes(const TileAttentionArgs &a) {
  const DeviceInfo device = currentDevice();
  if (!device.valid)
    return 0;
  size_t bytes = 0;
  visitTile(a, [&](auto shape) {
    TilePlan plan;
    if (planTile<decltype(shape)>(a, device, plan) == 0)
      bytes = plan.scratchBytes();
    return 0;
  });
  return bytes;
}

} // namespace token_major_gqa
