/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The packed layouts and decode formulas follow ggml-common.h and
// ggml-quants.c from llama.cpp commit b21e4de74567f5eef213765c9476a843c2e43f0d.
// llama.cpp is Copyright (c) 2023-2026 The ggml authors under the MIT License.

#include "hip_custom_kernels.h"
#include "kquant_abi.h"
#include "kquant_swiglu_device.h"
#include "kquant_dot.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

constexpr int64_t kBlockElements = HIP_KQUANT_BLOCK_ELEMENTS;
constexpr int64_t kQ8BlockElements = HIP_Q8_1_BLOCK_ELEMENTS;
constexpr int kThreads = 256;
constexpr int kGfx1151MmvqThreads = 32;
constexpr int kFallbackMmvqThreads = 256;
constexpr int kCachedDeviceCount = 16;

using Q4KBlock = hip_kquant_q4_k_block_t;
using Q5KBlock = hip_kquant_q5_k_block_t;
using Q6KBlock = hip_kquant_q6_k_block_t;

// Decode MMVQ quantizes F32 activations in 32-value Q8_1 blocks. llama.cpp's
// CPU K-quant dot uses 256-value Q8_K blocks, so this path is not numerically
// equivalent to that CPU accumulation path.
using Q8_1Block = hip_q8_1_block_t;

template <typename Block>
inline constexpr int64_t kWeightBlockElements = kBlockElements;

template <typename Block>
inline constexpr int64_t kQ8GroupsPerWeightBlock =
    kWeightBlockElements<Block> / kQ8BlockElements;

__device__ float halfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(__half(raw));
}

__device__ uint16_t floatToHalfBits(float value) {
  const __half converted = __float2half_rn(value);
  const __half_raw raw = static_cast<__half_raw>(converted);
  return raw.x;
}

__device__ void unpackQ4KScaleMin(int group, const uint8_t *packed,
                                  uint8_t &scale, uint8_t &minimum) {
  if (group < 4) {
    scale = packed[group] & 0x3f;
    minimum = packed[group + 4] & 0x3f;
    return;
  }
  scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
  minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
}

__device__ __forceinline__ int loadI32(const void *address) {
  int value;
  __builtin_memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ uint32_t loadU32(const void *address) {
  uint32_t value;
  __builtin_memcpy(&value, address, sizeof(value));
  return value;
}

__device__ int signedDot4(int lhs, int rhs, int accumulator) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__GFX11__) || defined(__GFX12__))
  return __builtin_amdgcn_sudot4(true, lhs, true, rhs, accumulator, false);
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
  const int8_t *lhsBytes = reinterpret_cast<const int8_t *>(&lhs);
  const int8_t *rhsBytes = reinterpret_cast<const int8_t *>(&rhs);
  for (int i = 0; i < 4; ++i) {
    accumulator += lhsBytes[i] * rhsBytes[i];
  }
  return accumulator;
#endif
}

__device__ __forceinline__ int packQ6KQuants(const Q6KBlock &block,
                                             int firstIndex) {
  const int half = firstIndex / 128;
  const int indexIn128 = firstIndex % 128;
  const int quarter = indexIn128 / 32;
  const int lane = indexIn128 % 32;

  const int lowIndex = half * 64 + lane + ((quarter & 1) ? 32 : 0);
  const uint32_t lowPacked = loadU32(block.ql + lowIndex);
  const uint32_t low = (lowPacked >> (quarter < 2 ? 0 : 4)) & 0x0f0f0f0fU;
  const uint32_t highPacked = loadU32(block.qh + half * 32 + lane);
  const uint32_t high = ((highPacked >> (2 * quarter)) & 0x03030303U) << 4;
  const uint32_t unsignedQuants = low | high;

  // Convert four independent bytes from [0, 63] to signed [-32, 31]
  // without a packed integer subtraction that could borrow across bytes.
  const uint32_t negative = (~unsignedQuants) & 0x20202020U;
  const uint32_t signedQuants = (unsignedQuants & 0x1f1f1f1fU) | negative |
                                (negative << 1) | (negative << 2);
  int result;
  __builtin_memcpy(&result, &signedQuants, sizeof(result));
  return result;
}

__device__ __forceinline__ int packQ5KQuants(const Q5KBlock &block, int group,
                                             int word) {
  const uint32_t packed =
      loadU32(block.qs + (group / 2) * 32 + word * sizeof(uint32_t));
  const uint32_t low =
      (group & 1) != 0 ? (packed >> 4) & 0x0f0f0f0fU : packed & 0x0f0f0f0fU;
  const uint32_t highBits = loadU32(block.qh + word * sizeof(uint32_t));
  const uint32_t high = ((highBits >> group) & 0x01010101U) << 4;
  const uint32_t quants = low | high;
  int result;
  __builtin_memcpy(&result, &quants, sizeof(result));
  return result;
}


__device__ float waveReduceSum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ float waveReduceMax(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return value;
}

__device__ int waveReduceSum(int value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    value += __shfl_down(value, offset);
  return value;
}

bool isValidQ8SumMode(int sumMode) {
  return sumMode == HIP_Q8_1_SUM_PREQUANTIZATION ||
         sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED ||
         sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM;
}

bool isValidQ8SumModeForFormat(int sumMode, int formatId) {
  if (!isValidQ8SumMode(sumMode))
    return false;
  (void)formatId;
  return true;
}

int resolveQ8SumMode(int sumMode, int64_t totalRows, int64_t rowsPerToken) {
  if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
      totalRows / rowsPerToken > 1) {
    return HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED;
  }
  return sumMode;
}

bool validQ8RowGrouping(int64_t M, int64_t batchCount, int64_t rowsPerToken,
                        int64_t &totalRows) {
  if (M <= 0 || batchCount <= 0 || rowsPerToken <= 0 ||
      M > std::numeric_limits<int64_t>::max() / batchCount) {
    return false;
  }
  totalRows = M * batchCount;
  return totalRows % rowsPerToken == 0;
}

__global__ void quantizeQ8_1Kernel(const float *__restrict__ activation,
                                   Q8_1Block *__restrict__ quantized,
                                   int64_t blockCount, int sumMode) {
  const int lane = threadIdx.x % warpSize;
  const int waveInBlock = threadIdx.x / warpSize;
  const int wavesPerBlock = blockDim.x / warpSize;
  const int64_t blockIndex =
      static_cast<int64_t>(blockIdx.x) * wavesPerBlock + waveInBlock;
  if (blockIndex >= blockCount) {
    return;
  }

  const float value = lane < 32 ? activation[blockIndex * 32 + lane] : 0.0f;
  float maximum = waveReduceMax(fabsf(value));
  maximum = __shfl(maximum, 0);
  const float scale = maximum / 127.0f;
  int quant = scale == 0.0f ? 0 : static_cast<int>(roundf(value / scale));
  quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
  float storedSum = 0.0f;
  int quantizedSum = 0;
  if (sumMode == HIP_Q8_1_SUM_PREQUANTIZATION)
    storedSum = waveReduceSum(value);
  else
    quantizedSum = waveReduceSum(quant);
  if (lane < 32) {
    quantized[blockIndex].qs[lane] = static_cast<int8_t>(quant);
  }
  if (lane == 0) {
    quantized[blockIndex].d = floatToHalfBits(scale);
    if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
      const int16_t signedSum = static_cast<int16_t>(quantizedSum);
      __builtin_memcpy(&quantized[blockIndex].s, &signedSum, sizeof(signedSum));
    } else {
      if (sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED)
        storedSum = scale * static_cast<float>(quantizedSum);
      quantized[blockIndex].s = floatToHalfBits(storedSum);
    }
  }
}

// A row per workgroup of Threads: as many waves as there are, each Q8 block
// of the row one wave's, so each wave quantizes K / 32 / waves of them one
// after another.
template <int Threads>
__global__ void addRmsNormQuantizeQ8_1Kernel(const float *lhs, const float *rhs,
                                             const float *scale,
                                             float *residual,
                                             Q8_1Block *quantized, int64_t K,
                                             float epsilon, int sumMode) {
  const int64_t row = blockIdx.x;
  const int64_t rowOffset = row * K;

  __shared__ float reduction[Threads];
  __shared__ float inverseRms;

  // Each thread keeps the sums it makes (columns threadIdx.x + i * blockDim.x)
  // for the quantization, which reads them back block by block: a wave's 32
  // lanes of step i are Q8 block i * waves + wave. Rows wider than that read
  // them back from `residual`.
  constexpr int kKeptSteps = 32;
  const int64_t steps = K / blockDim.x;
  const bool kept = K % blockDim.x == 0 && steps <= kKeptSteps &&
                    blockDim.x / warpSize * kQ8BlockElements == blockDim.x;
  float sums[kKeptSteps], scales[kKeptSteps];

  float localSumSquares = 0.0f;
  for (int64_t column = threadIdx.x, step = 0; column < K;
       column += blockDim.x, ++step) {
    const int64_t index = rowOffset + column;
    const float value = lhs[index] + rhs[index];
    residual[index] = value;
    localSumSquares += value * value;
    if (kept) {
      const float weight = scale[column];
#pragma unroll
      for (int i = 0; i < kKeptSteps; ++i)
        if (i == step) {
          sums[i] = value;
          scales[i] = weight;
        }
    }
  }

  reduction[threadIdx.x] = localSumSquares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    inverseRms = rsqrtf(reduction[0] / static_cast<float>(K) + epsilon);
  }
  __syncthreads();

  const int lane = threadIdx.x % warpSize;
  const int waveInBlock = threadIdx.x / warpSize;
  const int wavesPerBlock = blockDim.x / warpSize;
  const int64_t q8BlocksPerRow = K / kQ8BlockElements;
  for (int64_t q8Block = waveInBlock, step = 0; q8Block < q8BlocksPerRow;
       q8Block += wavesPerBlock, ++step) {
    const int64_t column = q8Block * kQ8BlockElements + lane;
    float value = 0.0f;
    if (lane < kQ8BlockElements) {
      float sum = 0.0f, weight = 0.0f;
      if (kept) {
#pragma unroll
        for (int i = 0; i < kKeptSteps; ++i)
          if (i == step) {
            sum = sums[i];
            weight = scales[i];
          }
      } else {
        sum = residual[rowOffset + column];
        weight = scale[column];
      }
      value = sum * inverseRms * weight;
    }

    float maximum = waveReduceMax(fabsf(value));
    maximum = __shfl(maximum, 0);
    const float q8Scale = maximum / 127.0f;
    int quant = q8Scale == 0.0f ? 0 : static_cast<int>(roundf(value / q8Scale));
    quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    float storedSum = 0.0f;
    int quantizedSum = 0;
    if (sumMode == HIP_Q8_1_SUM_PREQUANTIZATION)
      storedSum = waveReduceSum(value);
    else
      quantizedSum = waveReduceSum(quant);

    Q8_1Block &outputBlock = quantized[row * q8BlocksPerRow + q8Block];
    if (lane < kQ8BlockElements)
      outputBlock.qs[lane] = static_cast<int8_t>(quant);
    if (lane == 0) {
      outputBlock.d = floatToHalfBits(q8Scale);
      if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
        const int16_t signedSum = static_cast<int16_t>(quantizedSum);
        __builtin_memcpy(&outputBlock.s, &signedSum, sizeof(signedSum));
      } else {
        if (sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED)
          storedSum = q8Scale * static_cast<float>(quantizedSum);
        outputBlock.s = floatToHalfBits(storedSum);
      }
    }
  }
}

__device__ __forceinline__ void loadQ8_1(const Q8_1Block &activation,
                                         int (&words)[8], float &scale,
                                         float &sumPayload) {
#pragma unroll
  for (int word = 0; word < 8; ++word) {
    words[word] = loadI32(activation.qs + word * sizeof(int));
  }
  scale = halfBitsToFloat(activation.d);
  sumPayload = __uint_as_float(static_cast<uint32_t>(activation.s));
}

__device__ __forceinline__ uint16_t decodeQ8SumBits(float payload) {
  return static_cast<uint16_t>(__float_as_uint(payload));
}

__device__ __forceinline__ float decodeQ8StoredSum(float payload) {
  return halfBitsToFloat(decodeQ8SumBits(payload));
}

__device__ __forceinline__ int decodeQ8IntegerSum(float payload) {
  const uint16_t bits = decodeQ8SumBits(payload);
  int16_t sum;
  __builtin_memcpy(&sum, &bits, sizeof(sum));
  return sum;
}

template <bool UseIntegerSum>
__device__ __forceinline__ float
dotQ4KQ8_1Impl(const Q4KBlock &weight, int group, const int *activationWords,
               float activationScale, float activationSumPayload) {
  const int pair = group / 2;
  const bool highNibble = (group & 1) != 0;
  const uint8_t *weightBytes = weight.qs + static_cast<int64_t>(pair) * 32;
  int dot = 0;
  for (int word = 0; word < 8; ++word) {
    const uint32_t packed = loadU32(weightBytes + word * sizeof(uint32_t));
    const int quants = static_cast<int>(highNibble ? (packed >> 4) & 0x0f0f0f0fU
                                                   : packed & 0x0f0f0f0fU);
    dot = kquant_sudot4(activationWords[word], quants, dot);
  }

  uint8_t scale;
  uint8_t minimum;
  unpackQ4KScaleMin(group, weight.scales, scale, minimum);
  const float activationSum =
      UseIntegerSum
          ? activationScale *
                static_cast<float>(decodeQ8IntegerSum(activationSumPayload))
          : decodeQ8StoredSum(activationSumPayload);
  return halfBitsToFloat(weight.d) * activationScale *
             static_cast<float>(scale * dot) -
         halfBitsToFloat(weight.dmin) * static_cast<float>(minimum) *
             activationSum;
}

__device__ __forceinline__ float dotQ4KQ8_1(const Q4KBlock &weight, int group,
                                            const int *activationWords,
                                            float activationScale,
                                            float activationSumPayload) {
  return dotQ4KQ8_1Impl<false>(weight, group, activationWords, activationScale,
                               activationSumPayload);
}

__device__ __forceinline__ float
dotQ4KQ8_1IntegerSum(const Q4KBlock &weight, int group,
                     const int *activationWords, float activationScale,
                     float activationSumPayload) {
  return dotQ4KQ8_1Impl<true>(weight, group, activationWords, activationScale,
                              activationSumPayload);
}

template <bool UseIntegerSum>
__device__ __forceinline__ float
dotQ5KQ8_1Impl(const Q5KBlock &weight, int group, const int *activationWords,
               float activationScale, float activationSumPayload) {
  int dot = 0;
  for (int word = 0; word < 8; ++word) {
    dot = kquant_sudot4(activationWords[word],
                          packQ5KQuants(weight, group, word), dot);
  }
  uint8_t scale;
  uint8_t minimum;
  unpackQ4KScaleMin(group, weight.scales, scale, minimum);
  const float activationSum =
      UseIntegerSum
          ? activationScale *
                static_cast<float>(decodeQ8IntegerSum(activationSumPayload))
          : decodeQ8StoredSum(activationSumPayload);
  return halfBitsToFloat(weight.d) * activationScale *
             static_cast<float>(scale * dot) -
         halfBitsToFloat(weight.dmin) * static_cast<float>(minimum) *
             activationSum;
}

__device__ __forceinline__ float dotQ5KQ8_1(const Q5KBlock &weight, int group,
                                            const int *activationWords,
                                            float activationScale,
                                            float activationSumPayload) {
  return dotQ5KQ8_1Impl<false>(weight, group, activationWords, activationScale,
                               activationSumPayload);
}

__device__ __forceinline__ float
dotQ5KQ8_1IntegerSum(const Q5KBlock &weight, int group,
                     const int *activationWords, float activationScale,
                     float activationSumPayload) {
  return dotQ5KQ8_1Impl<true>(weight, group, activationWords, activationScale,
                              activationSumPayload);
}

__device__ __forceinline__ float dotQ6KQ8_1(const Q6KBlock &weight, int group,
                                            const int *activationWords,
                                            float activationScale, float) {
  int lowDot = 0;
  int highDot = 0;
  const int base = group * 32;
  for (int word = 0; word < 4; ++word) {
    lowDot = signedDot4(activationWords[word],
                        packQ6KQuants(weight, base + word * 4), lowDot);
    highDot = signedDot4(activationWords[word + 4],
                         packQ6KQuants(weight, base + 16 + word * 4), highDot);
  }
  const int scaleIndex = group * 2;
  const int scaledDot = weight.scales[scaleIndex] * lowDot +
                        weight.scales[scaleIndex + 1] * highDot;
  return halfBitsToFloat(weight.d) * activationScale *
         static_cast<float>(scaledDot);
}


template <int TileM>
__device__ void accumulateQ8Tile(const Q4KBlock &weight, int group,
                                 const Q8_1Block *activation,
                                 int64_t activationRowStride, int validRows,
                                 float (&sums)[TileM]) {
  int dots[TileM] = {};
  const int pair = group / 2;
  const bool highNibble = (group & 1) != 0;
  const uint8_t *weightBytes = weight.qs + static_cast<int64_t>(pair) * 32;
  for (int word = 0; word < 8; ++word) {
    const uint32_t packed = loadU32(weightBytes + word * sizeof(uint32_t));
    const int quants = static_cast<int>(highNibble ? (packed >> 4) & 0x0f0f0f0fU
                                                   : packed & 0x0f0f0f0fU);
#pragma unroll
    for (int row = 0; row < TileM; ++row) {
      if (row < validRows) {
        const int activationWord = loadI32(
            activation[row * activationRowStride].qs + word * sizeof(int));
        dots[row] = kquant_sudot4(activationWord, quants, dots[row]);
      }
    }
  }

  uint8_t scale;
  uint8_t minimum;
  unpackQ4KScaleMin(group, weight.scales, scale, minimum);
  const float weightScale = halfBitsToFloat(weight.d) * scale;
  const float weightMinimum = halfBitsToFloat(weight.dmin) * minimum;
#pragma unroll
  for (int row = 0; row < TileM; ++row) {
    if (row < validRows) {
      const float activationScale =
          halfBitsToFloat(activation[row * activationRowStride].d);
      const float activationSum =
          halfBitsToFloat(activation[row * activationRowStride].s);
      sums[row] += activationScale * weightScale * dots[row] -
                   weightMinimum * activationSum;
    }
  }
}

template <int TileM>
__device__ void accumulateQ8Tile(const Q5KBlock &weight, int group,
                                 const Q8_1Block *activation,
                                 int64_t activationRowStride, int validRows,
                                 float (&sums)[TileM]) {
  int dots[TileM] = {};
  for (int word = 0; word < 8; ++word) {
    const int quants = packQ5KQuants(weight, group, word);
#pragma unroll
    for (int row = 0; row < TileM; ++row) {
      if (row < validRows) {
        const int activationWord = loadI32(
            activation[row * activationRowStride].qs + word * sizeof(int));
        dots[row] = kquant_sudot4(activationWord, quants, dots[row]);
      }
    }
  }

  uint8_t scale;
  uint8_t minimum;
  unpackQ4KScaleMin(group, weight.scales, scale, minimum);
  const float weightScale = halfBitsToFloat(weight.d) * scale;
  const float weightMinimum = halfBitsToFloat(weight.dmin) * minimum;
#pragma unroll
  for (int row = 0; row < TileM; ++row) {
    if (row < validRows) {
      const float activationScale =
          halfBitsToFloat(activation[row * activationRowStride].d);
      const float activationSum =
          halfBitsToFloat(activation[row * activationRowStride].s);
      sums[row] += activationScale * weightScale * dots[row] -
                   weightMinimum * activationSum;
    }
  }
}

template <int TileM>
__device__ void accumulateQ8Tile(const Q6KBlock &weight, int group,
                                 const Q8_1Block *activation,
                                 int64_t activationRowStride, int validRows,
                                 float (&sums)[TileM]) {
  int lowDots[TileM] = {};
  int highDots[TileM] = {};
  const int base = group * 32;
  for (int word = 0; word < 4; ++word) {
    const int lowQuants = packQ6KQuants(weight, base + word * 4);
    const int highQuants = packQ6KQuants(weight, base + 16 + word * 4);
#pragma unroll
    for (int row = 0; row < TileM; ++row) {
      if (row < validRows) {
        const int8_t *activationBytes =
            activation[row * activationRowStride].qs;
        lowDots[row] = signedDot4(loadI32(activationBytes + word * sizeof(int)),
                                  lowQuants, lowDots[row]);
        highDots[row] =
            signedDot4(loadI32(activationBytes + (word + 4) * sizeof(int)),
                       highQuants, highDots[row]);
      }
    }
  }

  const int scaleIndex = group * 2;
  const float weightScale = halfBitsToFloat(weight.d);
#pragma unroll
  for (int row = 0; row < TileM; ++row) {
    if (row < validRows) {
      const int scaledDot = weight.scales[scaleIndex] * lowDots[row] +
                            weight.scales[scaleIndex + 1] * highDots[row];
      sums[row] += weightScale *
                   halfBitsToFloat(activation[row * activationRowStride].d) *
                   static_cast<float>(scaledDot);
    }
  }
}

// One wave per output column of one activation row.
template <typename Block,
          float (*Dot)(const Block &, int, const int *, float, float)>
__global__ void kquantMmvqKernel(const Q8_1Block *__restrict__ activation,
                                     const Block *__restrict__ packedWeights,
                                     float *__restrict__ output, int64_t N,
                                     int64_t K, int64_t taskCount) {
  const int lane = threadIdx.x % warpSize;
  const int waveInBlock = threadIdx.x / warpSize;
  const int wavesPerBlock = blockDim.x / warpSize;
  const int64_t task =
      static_cast<int64_t>(blockIdx.x) * wavesPerBlock + waveInBlock;
  if (task >= taskCount) {
    return;
  }

  const int64_t outputColumn = task % N;
  const int64_t activationRow = task / N;
  const int64_t activationBlocksPerRow = K / kQ8BlockElements;
  constexpr int64_t groupsPerWeightBlock = kQ8GroupsPerWeightBlock<Block>;
  const int64_t weightBlocksPerRow = K / kWeightBlockElements<Block>;
  const Q8_1Block *activationData =
      activation + activationRow * activationBlocksPerRow;
  const Block *weightData = packedWeights + outputColumn * weightBlocksPerRow;

  float sum = 0.0f;
  for (int64_t group = lane; group < activationBlocksPerRow;
       group += warpSize) {
    int activationWords[8];
    float activationScale;
    float activationSum;
    loadQ8_1(activationData[group], activationWords, activationScale,
             activationSum);
    sum += Dot(weightData[group / groupsPerWeightBlock],
               group % groupsPerWeightBlock, activationWords, activationScale,
               activationSum);
  }
  sum = waveReduceSum(sum);
  if (lane == 0) {
    output[activationRow * N + outputColumn] = sum;
  }
}

//===----------------------------------------------------------------------===//
// A few rows: llama.cpp's MMVQ (mmvq.cu mul_mat_vec_q with ncols_dst > 1, and
// the vec_dot_*_q8_1 of vecdotq.cuh) for the K-quants, as it runs on RDNA 3.5
// (one wave a block, one output column a block). Each thread reads `vdr`
// 32-bit words of a weight block and their activations, column after column:
// the weights stream from memory once for every row. A verify step's rows and
// a DFlash draft's block take it.
//===----------------------------------------------------------------------===//

template <typename Block> struct MmvqColumnsTraits;
template <> struct MmvqColumnsTraits<Q4KBlock> {
  static constexpr int qi = 32, vdr = 2;
};
template <> struct MmvqColumnsTraits<Q5KBlock> {
  static constexpr int qi = 32, vdr = 2;
};
template <> struct MmvqColumnsTraits<Q6KBlock> {
  static constexpr int qi = 32, vdr = 2;
};


/// An activation block's scale (ds.x).
__device__ __forceinline__ float q8Scale(const Q8_1Block &block) {
  return halfBitsToFloat(block.d);
}

/// The scale and minimum bytes of Q4_K / Q5_K sub-blocks 2j, 2j + 1.
__device__ __forceinline__ void columnsScaleMin(const uint8_t *packed,
                                                int bq8Offset, uint8_t (&sc)[2],
                                                uint8_t (&m)[2]) {
  const int j = bq8Offset / 2;
  const int jm = j & 1;
  auto half = [&](int i) {
    uint16_t value;
    __builtin_memcpy(&value, packed + 2 * i, sizeof(value));
    return static_cast<uint32_t>(value);
  };
  const uint32_t s0 = half(jm + 0), s2 = half(jm + 2), s4 = half(jm + 4);
  const uint32_t hi = static_cast<uint32_t>(-static_cast<int32_t>(j >= 2));
  const uint16_t aux0 = static_cast<uint16_t>(
      ((s0 & 0x3f3f) & ~hi) |
      ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
  const uint16_t aux1 = static_cast<uint16_t>(
      ((s2 & 0x3f3f) & ~hi) |
      ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
  sc[0] = static_cast<uint8_t>(aux0);
  sc[1] = static_cast<uint8_t>(aux0 >> 8);
  m[0] = static_cast<uint8_t>(aux1);
  m[1] = static_cast<uint8_t>(aux1 >> 8);
}

// vec_dot_q4_K_q8_1
__device__ __forceinline__ float vecDotColumns(const Q4KBlock &w,
                                               const Q8_1Block *y, int iqs) {
  constexpr int qr = 2;
  const int bq8Offset = qr * ((iqs / 2) / 4);
  const uint8_t *q4 = w.qs + 16 * bq8Offset + 4 * ((iqs / 2) % 4);
  const int v0 = loadI32(q4), v1 = loadI32(q4 + 16);
  uint8_t sc[2], m[2];
  columnsScaleMin(w.scales, bq8Offset, sc, m);
  float sumD = 0.0f, sumM = 0.0f;
#pragma unroll
  for (int i = 0; i < qr; ++i) {
    const Q8_1Block &b = y[bq8Offset + i];
    const int8_t *q8 = b.qs + 4 * ((iqs / 2) % 4);
    const int u0 = loadI32(q8), u1 = loadI32(q8 + 16);
    const int dot1 =
        signedDot4((v1 >> (4 * i)) & 0x0F0F0F0F, u1,
                   signedDot4((v0 >> (4 * i)) & 0x0F0F0F0F, u0, 0));
    const int dot2 = signedDot4(0x01010101, u1, signedDot4(0x01010101, u0, 0));
    const float d8 = q8Scale(b);
    sumD += d8 * (dot1 * sc[i]);
    sumM += d8 * (dot2 * m[i]);
  }
  return halfBitsToFloat(w.d) * sumD - halfBitsToFloat(w.dmin) * sumM;
}

// vec_dot_q5_K_q8_1
__device__ __forceinline__ float vecDotColumns(const Q5KBlock &w,
                                               const Q8_1Block *y, int iqs) {
  constexpr int qr = 2;
  const int bq8Offset = qr * ((iqs / 2) / 4);
  const uint8_t *ql = w.qs + 16 * bq8Offset + 4 * ((iqs / 2) % 4);
  const uint8_t *qh = w.qh + 4 * ((iqs / 2) % 4);
  const int vl0 = loadI32(ql), vl1 = loadI32(ql + 16);
  const int vh0 = loadI32(qh) >> bq8Offset, vh1 = loadI32(qh + 16) >> bq8Offset;
  uint8_t sc[2], m[2];
  columnsScaleMin(w.scales, bq8Offset, sc, m);
  float sumD = 0.0f, sumM = 0.0f;
#pragma unroll
  for (int i = 0; i < qr; ++i) {
    const Q8_1Block &b = y[bq8Offset + i];
    const int8_t *q8 = b.qs + 4 * ((iqs / 2) % 4);
    const int u0 = loadI32(q8), u1 = loadI32(q8 + 16);
    const int v0 =
        ((vl0 >> (4 * i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
    const int v1 =
        ((vl1 >> (4 * i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
    const int dot1 = signedDot4(v0, u0, signedDot4(v1, u1, 0));
    const int dot2 = signedDot4(0x01010101, u0, signedDot4(0x01010101, u1, 0));
    const float d8 = q8Scale(b);
    sumD += d8 * (dot1 * sc[i]);
    sumM += d8 * (dot2 * m[i]);
  }
  return halfBitsToFloat(w.d) * sumD - halfBitsToFloat(w.dmin) * sumM;
}

// vec_dot_q6_K_q8_1
__device__ __forceinline__ float vecDotColumns(const Q6KBlock &w,
                                               const Q8_1Block *y, int iqs) {
  constexpr int qr = 2, vdr = 2, qi = 32;
  const int bq8Offset = 2 * qr * (iqs / (qi / 2)) + (iqs % (qi / 2)) / (qi / 4);
  const int scaleOffset =
      (qi / 4) * (iqs / (qi / 2)) + (iqs % (qi / 2)) / (qi / 8);
  const int vhShift = 2 * ((iqs % (qi / 2)) / (qi / 4));
  int vl[vdr], vh[vdr];
#pragma unroll
  for (int j = 0; j < vdr; ++j) {
    vl[j] = loadI32(w.ql + 4 * (iqs + j));
    vh[j] = loadI32(w.qh +
                    4 * ((qi / 4) * (iqs / (qi / 2)) + iqs % (qi / 4) + j)) >>
            vhShift;
  }
  const int8_t *scales = w.scales + scaleOffset;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < qr; ++i) {
    const Q8_1Block &b = y[bq8Offset + 2 * i];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < vdr; ++j) {
      const int u = loadI32(b.qs + 4 * (iqs % 8 + j));
      const int vil = (vl[j] >> (4 * i)) & 0x0F0F0F0F;
      const int vih = ((vh[j] >> (4 * i)) << 4) & 0x30303030;
      // (vil | vih) - 32 for each byte; the - 32 is the second product
      sumi = signedDot4(vil | vih, u, sumi);
      sumi = signedDot4(static_cast<int>(0xE0E0E0E0u), u, sumi);
    }
    sum += q8Scale(b) * (sumi * scales[4 * i]);
  }
  return halfBitsToFloat(w.d) * sum;
}

template <typename Block, int Columns>
__global__ void __launch_bounds__(32, 1)
    kquantMmvqColumnsKernel(const Q8_1Block *__restrict__ activation,
                                const Block *__restrict__ packedWeights,
                                float *__restrict__ output, int64_t N,
                                int64_t K) {
  constexpr int qi = MmvqColumnsTraits<Block>::qi;
  constexpr int vdr = MmvqColumnsTraits<Block>::vdr;
  constexpr int warp = 32;
  constexpr int blocksPerIter = vdr * warp / qi;
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const int64_t batch = blockIdx.y;
  const int blocksPerRow = static_cast<int>(K / kWeightBlockElements<Block>);
  const int64_t q8PerRow = K / kQ8BlockElements;
  const Block *weights = packedWeights + row * blocksPerRow;
  const Q8_1Block *y = activation + batch * Columns * q8PerRow;

  float sums[Columns] = {};
  for (int kbx = tid / (qi / vdr); kbx < blocksPerRow; kbx += blocksPerIter) {
    const int kby = kbx * (kWeightBlockElements<Block> / kQ8BlockElements);
    const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
    for (int j = 0; j < Columns; ++j)
      sums[j] += vecDotColumns(weights[kbx], y + j * q8PerRow + kby, kqs);
  }
#pragma unroll
  for (int j = 0; j < Columns; ++j)
    sums[j] = waveReduceSum(sums[j]);
  if (tid == 0) {
#pragma unroll
    for (int j = 0; j < Columns; ++j)
      output[(batch * Columns + j) * N + row] = sums[j];
  }
}

template <typename Block>
int launchMmvqColumns(void *stream, const Q8_1Block *activation,
                      const void *packedWeights, float *output, int64_t M,
                      int64_t N, int64_t K, int64_t batchCount) {
  if (N > std::numeric_limits<unsigned>::max() || batchCount > 65535)
    return -1;
  const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(batchCount));
  const auto *weights = static_cast<const Block *>(packedWeights);
  auto *s = static_cast<hipStream_t>(stream);
  switch (M) {
#define HIP_KQUANT_COLUMNS(C)                                                    \
  case C:                                                                      \
    hipLaunchKernelGGL((kquantMmvqColumnsKernel<Block, C>), grid,          \
                       dim3(32), 0, s, activation, weights, output, N, K);     \
    break;
    HIP_KQUANT_COLUMNS(2)
    HIP_KQUANT_COLUMNS(3)
    HIP_KQUANT_COLUMNS(4)
    HIP_KQUANT_COLUMNS(5)
    HIP_KQUANT_COLUMNS(6)
    HIP_KQUANT_COLUMNS(7)
    HIP_KQUANT_COLUMNS(8)
#undef HIP_KQUANT_COLUMNS
  default:
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

template <typename GateBlock, typename UpBlock,
          float (*GateDot)(const GateBlock &, int, const int *, float, float),
          float (*UpDot)(const UpBlock &, int, const int *, float, float)>
__global__ void kquantMmvqSwiGluKernel(const Q8_1Block *activation,
                                           const GateBlock *gateWeights,
                                           const UpBlock *upWeights,
                                           float *output, int64_t N, int64_t K,
                                           int64_t taskCount) {
  const int lane = threadIdx.x % warpSize;
  const int waveInBlock = threadIdx.x / warpSize;
  const int wavesPerBlock = blockDim.x / warpSize;
  const int64_t task =
      static_cast<int64_t>(blockIdx.x) * wavesPerBlock + waveInBlock;
  if (task >= taskCount)
    return;

  const int64_t outputColumn = task % N;
  const int64_t activationRow = task / N;
  const int64_t activationBlocksPerRow = K / kQ8BlockElements;
  constexpr int64_t gateGroupsPerBlock = kQ8GroupsPerWeightBlock<GateBlock>;
  constexpr int64_t upGroupsPerBlock = kQ8GroupsPerWeightBlock<UpBlock>;
  const int64_t gateBlocksPerRow = K / kWeightBlockElements<GateBlock>;
  const int64_t upBlocksPerRow = K / kWeightBlockElements<UpBlock>;
  const Q8_1Block *activationData =
      activation + activationRow * activationBlocksPerRow;
  const GateBlock *gateWeightData =
      gateWeights + outputColumn * gateBlocksPerRow;
  const UpBlock *upWeightData = upWeights + outputColumn * upBlocksPerRow;

  float gateSum = 0.0f;
  float upSum = 0.0f;
  for (int64_t group = lane; group < activationBlocksPerRow;
       group += warpSize) {
    const Q8_1Block &activationBlock = activationData[group];
    int activationWords[8];
    float activationScale;
    float activationSum;
    loadQ8_1(activationBlock, activationWords, activationScale, activationSum);
    gateSum += GateDot(gateWeightData[group / gateGroupsPerBlock],
                       group % gateGroupsPerBlock, activationWords,
                       activationScale, activationSum);
    upSum +=
        UpDot(upWeightData[group / upGroupsPerBlock], group % upGroupsPerBlock,
              activationWords, activationScale, activationSum);
  }
  gateSum = waveReduceSum(gateSum);
  upSum = waveReduceSum(upSum);
  if (lane == 0)
    output[activationRow * N + outputColumn] =
        kquant_swiglu_value(gateSum, upSum);
}

template <typename Block, int TileM>
__global__ void
kquantMmvqTileKernel(const Q8_1Block *__restrict__ activation,
                         const Block *__restrict__ packedWeights,
                         float *__restrict__ output, int64_t M, int64_t N,
                         int64_t K, int64_t mTiles, int64_t taskCount) {
  const int lane = threadIdx.x % warpSize;
  const int waveInBlock = threadIdx.x / warpSize;
  const int wavesPerBlock = blockDim.x / warpSize;
  const int64_t task =
      static_cast<int64_t>(blockIdx.x) * wavesPerBlock + waveInBlock;
  if (task >= taskCount) {
    return;
  }

  const int64_t outputColumn = task % N;
  const int64_t tileAndBatch = task / N;
  const int64_t mTile = tileAndBatch % mTiles;
  const int64_t batch = tileAndBatch / mTiles;
  const int64_t mBase = mTile * TileM;
  const int64_t remainingRows = M - mBase;
  const int validRows =
      static_cast<int>(remainingRows < TileM ? remainingRows : TileM);
  const int64_t activationBlocksPerRow = K / kQ8BlockElements;
  constexpr int64_t groupsPerWeightBlock = kQ8GroupsPerWeightBlock<Block>;
  const int64_t weightBlocksPerRow = K / kWeightBlockElements<Block>;
  const Q8_1Block *activationData =
      activation + (batch * M + mBase) * activationBlocksPerRow;
  const Block *weightData = packedWeights + outputColumn * weightBlocksPerRow;

  float sums[TileM] = {};
  for (int64_t group = lane; group < activationBlocksPerRow;
       group += warpSize) {
    accumulateQ8Tile<TileM>(
        weightData[group / groupsPerWeightBlock], group % groupsPerWeightBlock,
        activationData + group, activationBlocksPerRow, validRows, sums);
  }
#pragma unroll
  for (int row = 0; row < TileM; ++row) {
    sums[row] = waveReduceSum(sums[row]);
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < TileM; ++row) {
      if (row < validRows) {
        output[((batch * M + mBase + row) * N) + outputColumn] = sums[row];
      }
    }
  }
}

bool positiveProductFits(int64_t lhs, int64_t rhs) {
  return lhs > 0 && rhs > 0 && lhs <= std::numeric_limits<int64_t>::max() / rhs;
}

int64_t formatBlockElements(int formatId) {
  switch (formatId) {
  case HIP_KQUANT_FORMAT_Q4_K:
  case HIP_KQUANT_FORMAT_Q5_K:
  case HIP_KQUANT_FORMAT_Q6_K:
    return kBlockElements;
  default:
    return 0;
  }
}

struct DeviceLaunchInfo {
  int waveSize;
  int defaultMmvqThreads;
};

constexpr bool isGfx1151(const char *architecture) {
  return architecture && architecture[0] == 'g' && architecture[1] == 'f' &&
         architecture[2] == 'x' && architecture[3] == '1' &&
         architecture[4] == '1' && architecture[5] == '5' &&
         architecture[6] == '1' &&
         (architecture[7] == '\0' || architecture[7] == ':');
}

constexpr int defaultMmvqThreads(const char *architecture) {
  return isGfx1151(architecture) ? kGfx1151MmvqThreads : kFallbackMmvqThreads;
}

static_assert(defaultMmvqThreads("gfx1151") == 32);
static_assert(defaultMmvqThreads("gfx1151:sramecc+:xnack-") == 32);
static_assert(defaultMmvqThreads("gfx11510") == 256);
static_assert(defaultMmvqThreads("gfx1100") == 256);

bool getDeviceLaunchInfo(DeviceLaunchInfo &info) {
  // Pack both values into one atomic entry so readers never observe a wave
  // size and architecture default from different devices or cache updates.
  static std::atomic<uint32_t> cache[kCachedDeviceCount]{};
  constexpr uint32_t waveMask = 0xffffU;

  int device = 0;
  if (hipGetDevice(&device) != hipSuccess || device < 0) {
    return false;
  }
  if (device < kCachedDeviceCount) {
    const uint32_t cached = cache[device].load(std::memory_order_acquire);
    if (cached != 0) {
      info.waveSize = static_cast<int>(cached & waveMask);
      info.defaultMmvqThreads = static_cast<int>(cached >> 16);
      return true;
    }
  }

  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return false;
  }
  info.waveSize = properties.warpSize;
  info.defaultMmvqThreads = defaultMmvqThreads(properties.gcnArchName);
  if (device < kCachedDeviceCount) {
    const uint32_t encoded =
        static_cast<uint32_t>(info.waveSize) |
        (static_cast<uint32_t>(info.defaultMmvqThreads) << 16);
    cache[device].store(encoded, std::memory_order_release);
  }
  return true;
}

int deviceWaveSize() {
  DeviceLaunchInfo info{};
  return getDeviceLaunchInfo(info) ? info.waveSize : -1;
}

int validatedMmvqThreads(int waveSize, int threads) {
  if (threads < waveSize || threads % waveSize != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: thread count %d "
                 "must be at least the device wave size %d and divisible by "
                 "it\n",
                 threads, waveSize);
    return -1;
  }
  return threads;
}

bool q8ScratchSize(int64_t M, int64_t K, int64_t batchCount, size_t &bytes) {
  const int64_t blocksPerRow = K / kQ8BlockElements;
  if (!positiveProductFits(M, K) || !positiveProductFits(M * K, batchCount) ||
      !positiveProductFits(M, batchCount) ||
      !positiveProductFits(M * batchCount, blocksPerRow)) {
    return false;
  }
  const uint64_t blockCount =
      static_cast<uint64_t>(M * batchCount * blocksPerRow);
  if (blockCount > std::numeric_limits<size_t>::max() / sizeof(Q8_1Block)) {
    return false;
  }
  bytes = static_cast<size_t>(blockCount) * sizeof(Q8_1Block);
  return true;
}

int launchQuantizeQ8_1(void *stream, const float *activation,
                       Q8_1Block *quantizedActivation, int64_t M, int64_t K,
                       int64_t batchCount, int waveSize, int sumMode,
                       int64_t rowsPerToken) {
  const int wavesPerBlock = kThreads / waveSize;
  const int64_t activationBlockCount = batchCount * M * (K / kQ8BlockElements);
  const int64_t quantizeGrid = activationBlockCount / wavesPerBlock +
                               (activationBlockCount % wavesPerBlock != 0);
  if (quantizeGrid >
      static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "[custom_kernels] hip_quantize_q8_1: grid "
                         "overflow\n");
    return -1;
  }
  const int effectiveSumMode = resolveQ8SumMode(
      sumMode, activationBlockCount / (K / kQ8BlockElements), rowsPerToken);
  hipLaunchKernelGGL(
      quantizeQ8_1Kernel, dim3(static_cast<unsigned>(quantizeGrid)),
      dim3(kThreads), 0, static_cast<hipStream_t>(stream), activation,
      quantizedActivation, activationBlockCount, effectiveSumMode);
  return static_cast<int>(hipGetLastError());
}

template <typename Block> bool packedWeightSizeFits(int64_t N, int64_t K) {
  constexpr int64_t blockElements = kWeightBlockElements<Block>;
  if (K <= 0 || K % blockElements != 0)
    return false;
  const int64_t blocksPerRow = K / blockElements;
  if (!positiveProductFits(N, blocksPerRow)) {
    return false;
  }
  const uint64_t blockCount = static_cast<uint64_t>(N * blocksPerRow);
  return blockCount <= std::numeric_limits<size_t>::max() / sizeof(Block);
}

bool f32TensorSizesFit(int64_t M, int64_t N, int64_t K, int64_t batchCount) {
  if (!positiveProductFits(M, K) || !positiveProductFits(M * K, batchCount) ||
      !positiveProductFits(M, N) || !positiveProductFits(M * N, batchCount)) {
    return false;
  }
  const uint64_t activationElements = static_cast<uint64_t>(batchCount * M * K);
  const uint64_t outputElements = static_cast<uint64_t>(batchCount * M * N);
  const uint64_t maximumElements =
      std::numeric_limits<size_t>::max() / sizeof(float);
  return activationElements <= maximumElements &&
         outputElements <= maximumElements;
}

template <typename Block,
          float (*Dot)(const Block &, int, const int *, float, float)>
int launchMmvq(void *stream, const void *packedWeights, float *output,
               int64_t N, int64_t K, int64_t batchCount,
               const Q8_1Block *quantizedActivation, int waveSize,
               int threads) {
  constexpr int64_t blockElements = kWeightBlockElements<Block>;
  if (!positiveProductFits(batchCount, N) ||
      !positiveProductFits(N, K / blockElements)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: shape product "
                 "overflow\n");
    return -1;
  }
  const int wavesPerBlock = threads / waveSize;
  const int64_t taskCount = batchCount * N;
  const int64_t matmulGrid =
      taskCount / wavesPerBlock + (taskCount % wavesPerBlock != 0);
  if (matmulGrid > static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul: MMVQ grid "
                         "overflow\n");
    return -1;
  }
  hipLaunchKernelGGL((kquantMmvqKernel<Block, Dot>),
                     dim3(static_cast<unsigned>(matmulGrid)), dim3(threads), 0,
                     static_cast<hipStream_t>(stream), quantizedActivation,
                     static_cast<const Block *>(packedWeights), output, N, K,
                     taskCount);
  return static_cast<int>(hipGetLastError());
}

template <typename GateBlock, typename UpBlock,
          float (*GateDot)(const GateBlock &, int, const int *, float, float),
          float (*UpDot)(const UpBlock &, int, const int *, float, float)>
int launchMmvqSwiGlu(void *stream, const void *quantizedActivation,
                     const void *gateWeights, const void *upWeights,
                     float *output, int64_t N, int64_t K, int64_t batchCount,
                     int waveSize, int defaultThreads) {
  constexpr int64_t gateBlockElements = kWeightBlockElements<GateBlock>;
  constexpr int64_t upBlockElements = kWeightBlockElements<UpBlock>;
  const int threads = validatedMmvqThreads(waveSize, defaultThreads);
  if (threads <= 0 || !positiveProductFits(batchCount, N) || K <= 0 ||
      K % gateBlockElements != 0 || K % upBlockElements != 0 ||
      !positiveProductFits(N, K / gateBlockElements) ||
      !positiveProductFits(N, K / upBlockElements)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: invalid "
                 "launch geometry\n");
    return -1;
  }
  const int wavesPerBlock = threads / waveSize;
  const int64_t taskCount = batchCount * N;
  const int64_t grid =
      taskCount / wavesPerBlock + (taskCount % wavesPerBlock != 0);
  if (grid > static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_swiglu_q8_1: grid "
                         "overflow\n");
    return -1;
  }
  hipLaunchKernelGGL(
      (kquantMmvqSwiGluKernel<GateBlock, UpBlock, GateDot, UpDot>),
      dim3(static_cast<unsigned>(grid)), dim3(threads), 0,
      static_cast<hipStream_t>(stream),
      static_cast<const Q8_1Block *>(quantizedActivation),
      static_cast<const GateBlock *>(gateWeights),
      static_cast<const UpBlock *>(upWeights), output, N, K, taskCount);
  return static_cast<int>(hipGetLastError());
}

template <typename Block, int TileM>
int launchMmvqTile(void *stream, const void *packedWeights, float *output,
                   int64_t M, int64_t N, int64_t K, int64_t batchCount,
                   const Q8_1Block *quantizedActivation, int waveSize) {
  const int64_t mTiles = M / TileM + (M % TileM != 0);
  constexpr int64_t blockElements = kWeightBlockElements<Block>;
  if (!positiveProductFits(M, K) || !positiveProductFits(M * K, batchCount) ||
      !positiveProductFits(M, N) || !positiveProductFits(M * N, batchCount) ||
      !positiveProductFits(mTiles, N) ||
      !positiveProductFits(mTiles * N, batchCount) ||
      !positiveProductFits(N, K / blockElements)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul: Q8_1 tiled "
                         "shape product overflow\n");
    return -1;
  }

  const int wavesPerBlock = kThreads / waveSize;
  const int64_t taskCount = batchCount * mTiles * N;
  const int64_t matmulGrid =
      taskCount / wavesPerBlock + (taskCount % wavesPerBlock != 0);
  if (matmulGrid > static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul: Q8_1 tiled "
                         "matmul grid overflow\n");
    return -1;
  }
  hipLaunchKernelGGL((kquantMmvqTileKernel<Block, TileM>),
                     dim3(static_cast<unsigned>(matmulGrid)), dim3(kThreads), 0,
                     static_cast<hipStream_t>(stream), quantizedActivation,
                     static_cast<const Block *>(packedWeights), output, M, N, K,
                     mTiles, taskCount);
  return static_cast<int>(hipGetLastError());
}

template <typename Block>
int launchSelectedMmvqTile(int tileM, void *stream, const void *packedWeights,
                           float *output, int64_t M, int64_t N, int64_t K,
                           int64_t batchCount,
                           const Q8_1Block *quantizedActivation, int waveSize) {
  switch (tileM) {
  case 4:
    return launchMmvqTile<Block, 4>(stream, packedWeights, output, M, N, K,
                                    batchCount, quantizedActivation, waveSize);
  case 8:
    return launchMmvqTile<Block, 8>(stream, packedWeights, output, M, N, K,
                                    batchCount, quantizedActivation, waveSize);
  case 16:
    return launchMmvqTile<Block, 16>(stream, packedWeights, output, M, N, K,
                                     batchCount, quantizedActivation, waveSize);
  default:
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: unsupported "
                 "Q8_1 row tile %d\n",
                 tileM);
    return -1;
  }
}

template <typename Block,
          float (*Dot)(const Block &, int, const int *, float, float)>
int launchPreparedQ8(void *stream, const Q8_1Block *quantizedActivation,
                     const void *packedWeights, float *output, int64_t M,
                     int64_t N, int64_t K, int64_t batchCount, int waveSize,
                     int defaultMmvqThreads) {
  if (M == 1) {
    // One wave per output column.
    const int threads = validatedMmvqThreads(waveSize, defaultMmvqThreads);
    if (threads <= 0)
      return -1;
    return launchMmvq<Block, Dot>(stream, packedWeights, output, N, K,
                                  batchCount, quantizedActivation, waveSize,
                                  threads);
  }

  // Up to eight rows as llama.cpp's MMVQ reads them.
  if (M <= 8 && waveSize == 32)
    return launchMmvqColumns<Block>(stream, quantizedActivation, packedWeights,
                                    output, M, N, K, batchCount);
  int selectedTile = 16;
  if (M <= 4)
    selectedTile = 4;
  else if (M <= 8 || std::is_same_v<Block, Q4KBlock>)
    selectedTile = 8;
  return launchSelectedMmvqTile<Block>(selectedTile, stream, packedWeights,
                                       output, M, N, K, batchCount,
                                       quantizedActivation, waveSize);
}

template <typename Block,
          float (*Dot)(const Block &, int, const int *, float, float)>
int launch(void *stream, const float *activation, const void *packedWeights,
           float *output, int64_t M, int64_t N, int64_t K, int64_t batchCount,
           int sumMode, void *q8Scratch, size_t q8ScratchBytes) {
  if (reinterpret_cast<uintptr_t>(packedWeights) % alignof(Block) != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: packed weights "
                 "are not aligned for the selected format\n");
    return -1;
  }
  if (!packedWeightSizeFits<Block>(N, K)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: packed weight "
                 "byte size overflow\n");
    return -1;
  }
  if (!f32TensorSizesFit(M, N, K, batchCount)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: F32 tensor byte "
                 "size overflow\n");
    return -1;
  }
  size_t requiredBytes = 0;
  DeviceLaunchInfo deviceInfo{};
  if (!q8ScratchSize(M, K, batchCount, requiredBytes)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul: Q8_1 "
                         "scratch size overflow\n");
    return -1;
  }
  if (!q8Scratch ||
      reinterpret_cast<uintptr_t>(q8Scratch) % alignof(Q8_1Block) != 0 ||
      q8ScratchBytes < requiredBytes) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: invalid Q8_1 "
                 "scratch buffer\n");
    return -1;
  }
  if (!getDeviceLaunchInfo(deviceInfo)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul: cannot "
                         "determine device wave size\n");
    return -1;
  }
  const int waveSize = deviceInfo.waveSize;
  if ((waveSize != 32 && waveSize != 64) || kThreads % waveSize != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: Q8_1 path "
                 "does not support wave size %d\n",
                 waveSize);
    return -1;
  }
  const int quantizeStatus = launchQuantizeQ8_1(
      stream, activation, static_cast<Q8_1Block *>(q8Scratch), M, K,
      batchCount, waveSize, sumMode,
      /*rowsPerToken=*/1);
  if (quantizeStatus != 0)
    return quantizeStatus;
  return launchPreparedQ8<Block, Dot>(
      stream, static_cast<const Q8_1Block *>(q8Scratch), packedWeights,
      output, M, N, K, batchCount, waveSize,
      deviceInfo.defaultMmvqThreads);
}

template <typename Block,
          float (*Dot)(const Block &, int, const int *, float, float)>
int launchPrepared(void *stream, const void *quantizedActivation,
                   const void *packedWeights, float *output, int64_t M,
                   int64_t N, int64_t K, int64_t batchCount) {
  if (reinterpret_cast<uintptr_t>(quantizedActivation) % alignof(Q8_1Block) !=
      0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: prepared "
                 "activation is not Q8_1-aligned\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(packedWeights) % alignof(Block) != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: packed "
                 "weights are not aligned for the selected format\n");
    return -1;
  }
  if (!packedWeightSizeFits<Block>(N, K) ||
      !f32TensorSizesFit(M, N, K, batchCount)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: tensor "
                 "byte size overflow\n");
    return -1;
  }
  size_t ignoredBytes = 0;
  if (!q8ScratchSize(M, K, batchCount, ignoredBytes)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_matmul_q8_1: Q8_1 "
                         "byte size overflow\n");
    return -1;
  }

  DeviceLaunchInfo deviceInfo{};
  if (!getDeviceLaunchInfo(deviceInfo)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: cannot "
                 "determine device properties\n");
    return -1;
  }
  const int waveSize = deviceInfo.waveSize;
  if ((waveSize != 32 && waveSize != 64) || kThreads % waveSize != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: unsupported "
                 "wave size %d\n",
                 waveSize);
    return -1;
  }
  return launchPreparedQ8<Block, Dot>(
      stream, static_cast<const Q8_1Block *>(quantizedActivation),
      packedWeights, output, M, N, K, batchCount, waveSize,
      deviceInfo.defaultMmvqThreads);
}

} // namespace

extern "C" int hip_quantize_q8_1(void *stream, const float *activation,
                                      void *quantizedActivation, int64_t M,
                                      int64_t K, int64_t batchCount,
                                      int sumMode, int64_t rowsPerToken) {
  if (!activation || !quantizedActivation) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_quantize_q8_1: null tensor\n");
    return -1;
  }
  int64_t totalRows = 0;
  if (!validQ8RowGrouping(M, batchCount, rowsPerToken, totalRows) || K <= 0 ||
      K % kQ8BlockElements != 0 || !isValidQ8SumMode(sumMode)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_quantize_q8_1: dimensions must "
                 "be positive, rows_per_token must divide the flattened row "
                 "count, K must be divisible by 32, and sum_mode must be "
                 "valid\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(quantizedActivation) % alignof(Q8_1Block) !=
      0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_quantize_q8_1: output is not "
                 "Q8_1-aligned\n");
    return -1;
  }
  size_t ignoredBytes = 0;
  if (!q8ScratchSize(M, K, batchCount, ignoredBytes)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_quantize_q8_1: Q8_1 byte size "
                 "overflow\n");
    return -1;
  }
  const int waveSize = deviceWaveSize();
  if ((waveSize != 32 && waveSize != 64) || kThreads % waveSize != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_quantize_q8_1: unsupported wave "
                 "size %d\n",
                 waveSize);
    return -1;
  }
  return launchQuantizeQ8_1(stream, activation,
                            static_cast<Q8_1Block *>(quantizedActivation), M, K,
                            batchCount, waveSize, sumMode, rowsPerToken);
}

extern "C" int hip_skip_rms_norm_q8_1(
    void *stream, const float *lhs, const float *rhs, const float *scale,
    float *residual, void *quantizedActivation, int64_t rowCount, int64_t K,
    float epsilon, int sumMode, int64_t rowsPerToken) {
  if (!lhs || !rhs || !scale || !residual || !quantizedActivation) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1: null tensor\n");
    return -1;
  }
  if (rowCount <= 0 || rowsPerToken <= 0 || rowCount % rowsPerToken != 0 ||
      K <= 0 || K % kBlockElements != 0 || !std::isfinite(epsilon) ||
      epsilon < 0.0f || !isValidQ8SumMode(sumMode)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1: dimensions "
                 "must be positive, rows_per_token must divide row_count, K "
                 "must be divisible by 256, epsilon must be finite and "
                 "non-negative, and sum_mode must be valid\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(quantizedActivation) % alignof(Q8_1Block) !=
      0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1: output is not "
                 "Q8_1-aligned\n");
    return -1;
  }
  size_t ignoredBytes = 0;
  if (!q8ScratchSize(rowCount, K, /*batchCount=*/1, ignoredBytes) ||
      rowCount > static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1: tensor size "
                 "or launch grid overflow\n");
    return -1;
  }
  const int waveSize = deviceWaveSize();
  if ((waveSize != 32 && waveSize != 64) || kThreads % waveSize != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1: unsupported "
                 "wave size %d\n",
                 waveSize);
    return -1;
  }

  const int effectiveSumMode =
      resolveQ8SumMode(sumMode, rowCount, rowsPerToken);
  // Few rows (a decode step's one) leave the device idle but for their
  // workgroups: four times the waves quantize a row's blocks in a quarter
  // the turns.
  constexpr int kFewRows = 64;
  if (rowCount <= kFewRows && K % (4 * kThreads) == 0)
    hipLaunchKernelGGL(addRmsNormQuantizeQ8_1Kernel<4 * kThreads>,
                       dim3(static_cast<unsigned>(rowCount)), dim3(4 * kThreads),
                       0, static_cast<hipStream_t>(stream), lhs, rhs, scale,
                       residual, static_cast<Q8_1Block *>(quantizedActivation),
                       K, epsilon, effectiveSumMode);
  else
    hipLaunchKernelGGL(addRmsNormQuantizeQ8_1Kernel<kThreads>,
                       dim3(static_cast<unsigned>(rowCount)), dim3(kThreads), 0,
                       static_cast<hipStream_t>(stream), lhs, rhs, scale,
                       residual, static_cast<Q8_1Block *>(quantizedActivation),
                       K, epsilon, effectiveSumMode);
  const hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_skip_rms_norm_q8_1 launch failed: "
                 "%s\n",
                 hipGetErrorString(error));
  }
  return static_cast<int>(error);
}

extern "C" int hip_kquant_matmul_q8_1(
    void *stream, const void *quantizedActivation, const void *packedWeights,
    float *output, int64_t M, int64_t N, int64_t K, int64_t batchCount,
    int formatId, int sumMode, int64_t rowsPerToken) {
  if (!quantizedActivation || !packedWeights || !output) {
    std::fprintf(
        stderr, "[custom_kernels] hip_kquant_matmul_q8_1: null tensor\n");
    return -1;
  }
  const int64_t blockElements = formatBlockElements(formatId);
  int64_t totalRows = 0;
  if (rowsPerToken != 1 ||
      !validQ8RowGrouping(M, batchCount, rowsPerToken, totalRows) || N <= 0 ||
      K <= 0 || blockElements <= 0 || K % blockElements != 0 ||
      !isValidQ8SumModeForFormat(sumMode, formatId) ||
      (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
       batchCount != 1)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: dimensions "
                 "must be positive and K must be divisible by the selected "
                 "format's block size, and sum_mode must be valid\n");
    return -1;
  }

  const int effectiveSumMode =
      resolveQ8SumMode(sumMode, totalRows, rowsPerToken);
  switch (formatId) {
  case HIP_KQUANT_FORMAT_Q4_K:
    if (effectiveSumMode ==
        HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
      return launchPrepared<Q4KBlock, dotQ4KQ8_1IntegerSum>(
          stream, quantizedActivation, packedWeights, output, M, N, K,
          batchCount);
    }
    return launchPrepared<Q4KBlock, dotQ4KQ8_1>(stream, quantizedActivation,
                                                packedWeights, output, M, N, K,
                                                batchCount);
  case HIP_KQUANT_FORMAT_Q5_K:
    if (effectiveSumMode ==
        HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
      return launchPrepared<Q5KBlock, dotQ5KQ8_1IntegerSum>(
          stream, quantizedActivation, packedWeights, output, M, N, K,
          batchCount);
    }
    return launchPrepared<Q5KBlock, dotQ5KQ8_1>(stream, quantizedActivation,
                                                packedWeights, output, M, N, K,
                                                batchCount);
  case HIP_KQUANT_FORMAT_Q6_K:
    return launchPrepared<Q6KBlock, dotQ6KQ8_1>(stream, quantizedActivation,
                                                packedWeights, output, M, N, K,
                                                batchCount);
  default:
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul_q8_1: unsupported "
                 "format_id=%d\n",
                 formatId);
    return -1;
  }
}

extern "C" int hip_kquant_swiglu_q8_1(
    void *stream, const void *quantizedActivation, const void *gateWeights,
    const void *upWeights, float *output, int64_t M, int64_t N, int64_t K,
    int64_t batchCount, int gateFormatId, int upFormatId, int sumMode,
    int64_t rowsPerToken, void *outputScratch, size_t outputScratchSize) {
  if (!quantizedActivation || !gateWeights || !upWeights || !output) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_swiglu_q8_1: null "
                         "required tensor\n");
    return -1;
  }
  int64_t totalRows = 0;
  if (rowsPerToken != 1 ||
      !validQ8RowGrouping(M, batchCount, rowsPerToken, totalRows) || N <= 0 ||
      K <= 0 || !isValidQ8SumModeForFormat(sumMode, gateFormatId) ||
      !isValidQ8SumModeForFormat(sumMode, upFormatId) ||
      (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
       batchCount != 1)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: dimensions "
                 "must be positive\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(output) % alignof(float) != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: output is "
                 "not float-aligned\n");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(quantizedActivation) % alignof(Q8_1Block) !=
      0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: prepared "
                 "activation is not Q8_1-aligned\n");
    return -1;
  }

  const auto hasFormatAlignment = [](const void *weights, int format) {
    if (format == HIP_KQUANT_FORMAT_Q4_K)
      return reinterpret_cast<uintptr_t>(weights) % alignof(Q4KBlock) == 0;
    if (format == HIP_KQUANT_FORMAT_Q6_K)
      return reinterpret_cast<uintptr_t>(weights) % alignof(Q6KBlock) == 0;
    return false;
  };
  if (!hasFormatAlignment(gateWeights, gateFormatId) ||
      !hasFormatAlignment(upWeights, upFormatId)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: unsupported "
                 "format or misaligned packed weights\n");
    return -1;
  }
  const auto packedSizeFits = [&](int format) {
    if (format == HIP_KQUANT_FORMAT_Q4_K)
      return packedWeightSizeFits<Q4KBlock>(N, K);
    if (format == HIP_KQUANT_FORMAT_Q6_K)
      return packedWeightSizeFits<Q6KBlock>(N, K);
    return false;
  };
  size_t ignoredQ8Bytes = 0;
  if (!packedSizeFits(gateFormatId) || !packedSizeFits(upFormatId) ||
      !f32TensorSizesFit(M, N, K, batchCount) ||
      !q8ScratchSize(M, K, batchCount, ignoredQ8Bytes)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: tensor "
                 "byte size overflow\n");
    return -1;
  }

  if (M == 1) {
    DeviceLaunchInfo deviceInfo{};
    if (!getDeviceLaunchInfo(deviceInfo) ||
        (deviceInfo.waveSize != 32 && deviceInfo.waveSize != 64) ||
        kThreads % deviceInfo.waveSize != 0) {
      std::fprintf(stderr, "[custom_kernels] hip_kquant_swiglu_q8_1: "
                           "unsupported device wave size\n");
      return -1;
    }
#define DISPATCH_SWIGLU(GateType, GateDot, UpType, UpDot)                      \
  return launchMmvqSwiGlu<GateType, UpType, GateDot, UpDot>(                   \
      stream, quantizedActivation, gateWeights, upWeights, output, N, K,       \
      batchCount, deviceInfo.waveSize, deviceInfo.defaultMmvqThreads)
    const bool useIntegerSum =
        resolveQ8SumMode(sumMode, totalRows, rowsPerToken) ==
        HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM;
    if (useIntegerSum && gateFormatId == HIP_KQUANT_FORMAT_Q4_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q4_K)
      DISPATCH_SWIGLU(Q4KBlock, dotQ4KQ8_1IntegerSum, Q4KBlock,
                      dotQ4KQ8_1IntegerSum);
    if (useIntegerSum && gateFormatId == HIP_KQUANT_FORMAT_Q4_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q6_K)
      DISPATCH_SWIGLU(Q4KBlock, dotQ4KQ8_1IntegerSum, Q6KBlock, dotQ6KQ8_1);
    if (useIntegerSum && gateFormatId == HIP_KQUANT_FORMAT_Q6_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q4_K)
      DISPATCH_SWIGLU(Q6KBlock, dotQ6KQ8_1, Q4KBlock, dotQ4KQ8_1IntegerSum);
    if (gateFormatId == HIP_KQUANT_FORMAT_Q4_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q4_K)
      DISPATCH_SWIGLU(Q4KBlock, dotQ4KQ8_1, Q4KBlock, dotQ4KQ8_1);
    if (gateFormatId == HIP_KQUANT_FORMAT_Q4_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q6_K)
      DISPATCH_SWIGLU(Q4KBlock, dotQ4KQ8_1, Q6KBlock, dotQ6KQ8_1);
    if (gateFormatId == HIP_KQUANT_FORMAT_Q6_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q4_K)
      DISPATCH_SWIGLU(Q6KBlock, dotQ6KQ8_1, Q4KBlock, dotQ4KQ8_1);
    if (gateFormatId == HIP_KQUANT_FORMAT_Q6_K &&
        upFormatId == HIP_KQUANT_FORMAT_Q6_K)
      DISPATCH_SWIGLU(Q6KBlock, dotQ6KQ8_1, Q6KBlock, dotQ6KQ8_1);
#undef DISPATCH_SWIGLU
    return -1;
  }

  if (!positiveProductFits(M, N) || !positiveProductFits(M * N, batchCount)) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_swiglu_q8_1: output "
                 "element count overflow\n");
    return -1;
  }
  const uint64_t outputElements = static_cast<uint64_t>(M * N * batchCount);
  if (outputElements > std::numeric_limits<size_t>::max() / sizeof(float) ||
      !outputScratch ||
      reinterpret_cast<uintptr_t>(outputScratch) % alignof(float) != 0 ||
      outputScratchSize < static_cast<size_t>(outputElements) * sizeof(float)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_swiglu_q8_1: M>1 "
                         "requires sufficient output scratch\n");
    return -1;
  }

  auto *gateOutput = static_cast<float *>(outputScratch);
  int status = hip_kquant_matmul_q8_1(
      stream, quantizedActivation, gateWeights, gateOutput, M, N, K, batchCount,
      gateFormatId, sumMode, rowsPerToken);
  if (status != 0)
    return status;
  status = hip_kquant_matmul_q8_1(stream, quantizedActivation, upWeights,
                                        output, M, N, K, batchCount, upFormatId,
                                        sumMode, rowsPerToken);
  if (status != 0)
    return status;
  return hip_swiglu(stream, gateOutput, output, output,
                    static_cast<int64_t>(outputElements), HIP_DTYPE_FLOAT32);
}

extern "C" int hip_kquant_matmul(void *stream, const float *activation,
                                       const void *packedWeights, float *output,
                                       int64_t M, int64_t N, int64_t K,
                                       int64_t batchCount, int formatId,
                                       void *q8Scratch,
                                       size_t q8ScratchBytes) {
  if (!activation || !packedWeights || !output) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: null tensor\n");
    return -1;
  }
  const int64_t blockElements = formatBlockElements(formatId);
  if (M <= 0 || N <= 0 || K <= 0 || batchCount <= 0 || blockElements <= 0 ||
      K % blockElements != 0) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: dimensions must "
                 "be positive and K must be divisible by the selected "
                 "format's block size\n");
    return -1;
  }

  switch (formatId) {
  case HIP_KQUANT_FORMAT_Q4_K:
    if (M == 1 && batchCount == 1) {
      return launch<Q4KBlock, dotQ4KQ8_1IntegerSum>(
          stream, activation, packedWeights, output, M, N, K, batchCount,
          HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM, q8Scratch,
          q8ScratchBytes);
    }
    return launch<Q4KBlock, dotQ4KQ8_1>(
        stream, activation, packedWeights, output, M, N, K, batchCount,
        HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM, q8Scratch,
        q8ScratchBytes);
  case HIP_KQUANT_FORMAT_Q5_K:
    return launch<Q5KBlock, dotQ5KQ8_1>(
        stream, activation, packedWeights, output, M, N, K, batchCount,
        HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED, q8Scratch, q8ScratchBytes);
  case HIP_KQUANT_FORMAT_Q6_K:
    return launch<Q6KBlock, dotQ6KQ8_1>(
        stream, activation, packedWeights, output, M, N, K, batchCount,
        HIP_Q8_1_SUM_PREQUANTIZATION, q8Scratch, q8ScratchBytes);
  default:
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_matmul: unsupported "
                 "format_id=%d\n",
                 formatId);
    return -1;
  }
}
