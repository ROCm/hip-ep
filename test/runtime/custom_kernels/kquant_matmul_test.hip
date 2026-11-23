/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "KQuantTestSupport.h"
#include "HipTestSupport.h"
#include "hip_custom_kernels.h"
#include "kquant_abi.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using hip::kquant::Q4KBlock;
using hip::kquant::Q6KBlock;
using hipdnn_ep::test::DeviceBuffer;
using hipdnn_ep::test::kquant::floatToHalfBits;
using hipdnn_ep::test::kquant::makeKQuantWeights;
using hipdnn_ep::test::kquant::Q8_1Block;
using hipdnn_ep::test::kquant::referenceDotQ8_1;

constexpr int64_t kQkK = hip::kquant::kKQuantBlockElements;
constexpr float kGuard = -9876.5f;
constexpr float kAbsoluteTolerance = 5.0e-2f;
constexpr float kRelativeTolerance = 2.0e-4f;
constexpr char kTestName[] = "kquant_matmul_test";

static_assert(sizeof(Q8_1Block) == HIP_Q8_1_BLOCK_SIZE);
static_assert(alignof(Q8_1Block) == HIP_Q8_1_BLOCK_ALIGNMENT);

bool checkHip(hipError_t error, const char *operation) {
  return hipdnn_ep::test::checkHip(error, operation, kTestName);
}

float wave32SumReference(const float *values) {
  std::array<float, hip::kquant::kQ8_1BlockElements> partial;
  std::copy_n(values, partial.size(), partial.begin());
  for (int offset = 16; offset > 0; offset >>= 1) {
    for (int lane = 0; lane < offset; ++lane) {
      partial[static_cast<size_t>(lane)] +=
          partial[static_cast<size_t>(lane + offset)];
    }
  }
  return partial[0];
}

std::vector<float> makeActivation(int64_t rows, int64_t columns) {
  std::vector<float> activation(static_cast<size_t>(rows * columns));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < columns; ++column) {
      const float phase = static_cast<float>((row + 5) * (column + 3));
      const float saw = static_cast<float>((column * 7 + row * 11) % 31 - 15);
      activation[static_cast<size_t>(row * columns + column)] =
          0.8f * std::cos(phase * 0.019f) + saw * 0.017f;
    }
  }
  return activation;
}

/// The Q8_1 blocks a producer writes in `sumMode`: mode 2 means its one-token
/// form, the integer sum bits.
std::vector<Q8_1Block>
quantizeQ8_1Reference(const std::vector<float> &activation,
                      int sumMode = HIP_Q8_1_SUM_PREQUANTIZATION) {
  if (activation.size() % hip::kquant::kQ8_1BlockElements != 0) {
    return {};
  }
  std::vector<Q8_1Block> blocks(activation.size() /
                                hip::kquant::kQ8_1BlockElements);
  for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
    const float *input =
        activation.data() + blockIndex * hip::kquant::kQ8_1BlockElements;
    float maximum = 0.0f;
    for (int i = 0; i < hip::kquant::kQ8_1BlockElements; ++i) {
      maximum = std::max(maximum, std::fabs(input[i]));
    }
    const float scale = maximum / 127.0f;
    for (int i = 0; i < hip::kquant::kQ8_1BlockElements; ++i) {
      int quant =
          scale == 0.0f ? 0 : static_cast<int>(std::round(input[i] / scale));
      quant = std::clamp(quant, -127, 127);
      blocks[blockIndex].qs[i] = static_cast<int8_t>(quant);
    }
    blocks[blockIndex].d = floatToHalfBits(scale);
    int16_t integerSum = 0;
    for (int8_t quant : blocks[blockIndex].qs)
      integerSum += quant;
    if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
      std::memcpy(&blocks[blockIndex].s, &integerSum, sizeof(integerSum));
    } else if (sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED) {
      blocks[blockIndex].s =
          floatToHalfBits(scale * static_cast<float>(integerSum));
    } else {
      blocks[blockIndex].s = floatToHalfBits(wave32SumReference(input));
    }
  }
  return blocks;
}

// This Q8_1 reference uses one scale per 32 values. llama.cpp's CPU K-quant
// dot path uses Q8_K with one scale per 256 values, so the two paths are not
// numerically equivalent even though both use eight-bit activations.
std::vector<float>
dequantizeQ8_1Reference(const std::vector<Q8_1Block> &blocks) {
  std::vector<float> result(blocks.size() * hip::kquant::kQ8_1BlockElements);
  for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
    const float scale = hip::kquant::fp16BitsToFloat(blocks[blockIndex].d);
    for (int i = 0; i < hip::kquant::kQ8_1BlockElements; ++i) {
      result[blockIndex * hip::kquant::kQ8_1BlockElements + i] =
          scale * static_cast<float>(blocks[blockIndex].qs[i]);
    }
  }
  return result;
}

struct MmqQ8_1Reference {
  std::vector<float> dequantized;
  std::vector<uint8_t> packed;
};

MmqQ8_1Reference quantizeMmqQ8_1Reference(const std::vector<float> &activation,
                                          int64_t M, int64_t K,
                                          int64_t batchCount) {
  constexpr int64_t blocksPerMmqBlock = HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE;
  constexpr int64_t valuesPerMmqBlock =
      blocksPerMmqBlock * hip::kquant::kQ8_1BlockElements;
  constexpr size_t metadataBytes = blocksPerMmqBlock * sizeof(float);
  constexpr size_t mmqBlockBytes = blocksPerMmqBlock * sizeof(Q8_1Block);
  if (M <= 0 || K <= 0 || batchCount <= 0 || K % valuesPerMmqBlock != 0 ||
      activation.size() != static_cast<size_t>(batchCount * M * K)) {
    return {};
  }

  MmqQ8_1Reference result;
  result.dequantized.resize(activation.size());
  result.packed.resize(
      static_cast<size_t>(batchCount * M * K / hip::kquant::kQ8_1BlockElements) *
      sizeof(Q8_1Block));
  const int64_t mmqBlocksPerRow = K / valuesPerMmqBlock;
  for (int64_t batch = 0; batch < batchCount; ++batch) {
    for (int64_t block = 0; block < mmqBlocksPerRow; ++block) {
      for (int64_t row = 0; row < M; ++row) {
        uint8_t *destination =
            result.packed.data() +
            ((batch * mmqBlocksPerRow + block) * M + row) * mmqBlockBytes;
        for (int64_t group = 0; group < blocksPerMmqBlock; ++group) {
          const size_t base = static_cast<size_t>(
              (batch * M + row) * K + block * valuesPerMmqBlock +
              group * hip::kquant::kQ8_1BlockElements);
          float maximum = 0.0f;
          for (int element = 0; element < hip::kquant::kQ8_1BlockElements;
               ++element) {
            maximum = std::max(maximum, std::fabs(activation[base + element]));
          }
          const float inverseScale = maximum == 0.0f ? 0.0f : 127.0f / maximum;
          const float scale = inverseScale == 0.0f ? 0.0f : 1.0f / inverseScale;
          std::memcpy(destination + group * sizeof(float), &scale,
                      sizeof(scale));
          for (int element = 0; element < hip::kquant::kQ8_1BlockElements;
               ++element) {
            const int quantized = static_cast<int>(
                std::round(activation[base + element] * inverseScale));
            result.dequantized[base + element] =
                scale * static_cast<float>(quantized);
            destination[metadataBytes + group * hip::kquant::kQ8_1BlockElements +
                        element] =
                static_cast<uint8_t>(static_cast<int8_t>(quantized));
          }
        }
      }
    }
  }
  return result;
}

std::vector<uint8_t> packMmqQ8_1Reference(const std::vector<Q8_1Block> &source,
                                          int64_t M, int64_t K,
                                          int64_t batchCount, bool storeSums) {
  constexpr int64_t blocksPerMmqBlock = HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE;
  constexpr size_t metadataBytes =
      HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE * sizeof(uint32_t);
  constexpr size_t mmqBlockBytes =
      HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE * sizeof(Q8_1Block);
  const int64_t q8BlocksPerRow = K / hip::kquant::kQ8_1BlockElements;
  const int64_t mmqBlocksPerRow = q8BlocksPerRow / blocksPerMmqBlock;
  std::vector<uint8_t> packed(source.size() * sizeof(Q8_1Block));
  for (int64_t batch = 0; batch < batchCount; ++batch) {
    for (int64_t block = 0; block < mmqBlocksPerRow; ++block) {
      uint8_t *destination =
          packed.data() + (batch * mmqBlocksPerRow + block) * M * mmqBlockBytes;
      for (int64_t row = 0; row < M; ++row) {
        uint8_t *destinationRow = destination + row * mmqBlockBytes;
        for (int64_t group = 0; group < blocksPerMmqBlock; ++group) {
          const Q8_1Block &sourceBlock =
              source[(batch * M + row) * q8BlocksPerRow +
                     block * blocksPerMmqBlock + group];
          if (storeSums) {
            std::memcpy(destinationRow + group * sizeof(uint32_t), &sourceBlock,
                        sizeof(uint32_t));
          } else {
            const float scale = hip::kquant::fp16BitsToFloat(sourceBlock.d);
            std::memcpy(destinationRow + group * sizeof(float), &scale,
                        sizeof(float));
          }
          std::memcpy(destinationRow + metadataBytes +
                          group * hip::kquant::kQ8_1BlockElements,
                      sourceBlock.qs, hip::kquant::kQ8_1BlockElements);
        }
      }
    }
  }
  return packed;
}

bool verifyLossyQ8_1(const std::vector<float> &input,
                     const std::vector<float> &dequantized, const char *name) {
  bool differs = false;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!std::isfinite(dequantized[i])) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s Q8_1 produced non-finite "
                   "value at %zu\n",
                   name, i);
      return false;
    }
    differs |= input[i] != dequantized[i];
  }
  if (!differs) {
    std::fprintf(stderr,
                 "ggml_k_quant_matvec_test: %s Q8_1 unexpectedly preserved "
                 "every activation exactly\n",
                 name);
  }
  return differs;
}

bool closeEnough(float actual, float expected) {
  // The GEMV path combines lane-local FMA sums with a wave reduction, while
  // the independent host oracle accumulates sequentially. Allow only the
  // expected FP32 reassociation error.
  return std::isfinite(actual) &&
         std::fabs(actual - expected) <=
             kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
}

template <typename Block>
bool copyAndRun(const std::vector<Block> &packed,
                const std::vector<float> &activation,
                const std::vector<float> &expected, int64_t rows,
                int64_t columns, int64_t M, int64_t batchCount,
                const char *name, int formatId,
                const std::vector<Q8_1Block> &expectedQ8) {
  constexpr size_t guardElements = 7;
  const int64_t outputElements = batchCount * M * rows;
  std::vector<float> actual(static_cast<size_t>(outputElements) + guardElements,
                            kGuard);
  DeviceBuffer weightsDevice(packed.size() * sizeof(Block));
  DeviceBuffer activationDevice(activation.size() * sizeof(float));
  DeviceBuffer outputDevice(actual.size() * sizeof(float));
  DeviceBuffer scratchDevice(
      std::max<size_t>(expectedQ8.size() * sizeof(Q8_1Block), 1));
  if (!weightsDevice.get() || !activationDevice.get() || !outputDevice.get() ||
      !scratchDevice.get()) {
    std::fprintf(stderr, "ggml_k_quant_matvec_test: %s allocation failed\n",
                 name);
    return false;
  }
  if (!checkHip(hipMemcpy(weightsDevice.get(), packed.data(),
                          packed.size() * sizeof(Block), hipMemcpyHostToDevice),
                "weights H2D") ||
      !checkHip(hipMemcpy(activationDevice.get(), activation.data(),
                          activation.size() * sizeof(float),
                          hipMemcpyHostToDevice),
                "activation H2D") ||
      !checkHip(hipMemcpy(outputDevice.get(), actual.data(),
                          actual.size() * sizeof(float), hipMemcpyHostToDevice),
                "output H2D")) {
    return false;
  }

  if (hip_kquant_matmul(
          nullptr, static_cast<const float *>(activationDevice.get()),
          weightsDevice.get(), static_cast<float *>(outputDevice.get()), M,
          rows, columns, batchCount, formatId,
          expectedQ8.empty() ? nullptr : scratchDevice.get(),
          expectedQ8.size() * sizeof(Q8_1Block)) != 0 ||
      !checkHip(hipMemcpy(actual.data(), outputDevice.get(),
                          actual.size() * sizeof(float), hipMemcpyDeviceToHost),
                "output D2H")) {
    return false;
  }

  if (!expectedQ8.empty()) {
    std::vector<Q8_1Block> actualQ8(expectedQ8.size());
    if (!checkHip(hipMemcpy(actualQ8.data(), scratchDevice.get(),
                            actualQ8.size() * sizeof(Q8_1Block),
                            hipMemcpyDeviceToHost),
                  "Q8_1 scratch D2H") ||
        std::memcmp(actualQ8.data(), expectedQ8.data(),
                    expectedQ8.size() * sizeof(Q8_1Block)) != 0) {
      size_t first = 0;
      while (first < actualQ8.size() &&
             std::memcmp(&actualQ8[first], &expectedQ8[first],
                         sizeof(Q8_1Block)) == 0)
        ++first;
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s Q8_1 scratch mismatch in "
                   "block %zu (d actual=%04x expected=%04x, s actual=%04x "
                   "expected=%04x)\n",
                   name, first, static_cast<unsigned>(actualQ8[first].d),
                   static_cast<unsigned>(expectedQ8[first].d),
                   static_cast<unsigned>(actualQ8[first].s),
                   static_cast<unsigned>(expectedQ8[first].s));
      return false;
    }
  }

  float maximumDifference = 0.0f;
  for (int64_t row = 0; row < outputElements; ++row) {
    maximumDifference = std::max(maximumDifference,
                                 std::fabs(actual[static_cast<size_t>(row)] -
                                           expected[static_cast<size_t>(row)]));
    if (!closeEnough(actual[static_cast<size_t>(row)],
                     expected[static_cast<size_t>(row)])) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s row %lld mismatch: "
                   "actual=%g expected=%g\n",
                   name, static_cast<long long>(row),
                   static_cast<double>(actual[static_cast<size_t>(row)]),
                   static_cast<double>(expected[static_cast<size_t>(row)]));
      return false;
    }
  }
  for (size_t index = static_cast<size_t>(outputElements);
       index < actual.size(); ++index) {
    if (actual[index] != kGuard) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s wrote output guard %zu\n",
                   name, index - static_cast<size_t>(outputElements));
      return false;
    }
  }

  // Exercise the split ABI used by compiler-generated shared activations. It
  // must produce byte-identical Q8_1 data and the same output as the combined
  // entry point.
  if (!expectedQ8.empty()) {
    const int sumMode =
        batchCount == 1 ? HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM
        : formatId == HIP_KQUANT_FORMAT_Q4_K
            ? HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED
            : HIP_Q8_1_SUM_PREQUANTIZATION;
    std::fill(actual.begin(), actual.end(), kGuard);
    if (!checkHip(hipMemcpy(outputDevice.get(), actual.data(),
                            actual.size() * sizeof(float),
                            hipMemcpyHostToDevice),
                  "prepared output reset") ||
        hip_quantize_q8_1(
            nullptr, static_cast<const float *>(activationDevice.get()),
            scratchDevice.get(), M, columns, batchCount, sumMode,
            /*rows_per_token=*/1) != 0 ||
        hip_kquant_matmul_q8_1(
            nullptr, scratchDevice.get(), weightsDevice.get(),
            static_cast<float *>(outputDevice.get()), M, rows, columns,
            batchCount, formatId, sumMode,
            /*rows_per_token=*/1) != 0 ||
        !checkHip(hipMemcpy(actual.data(), outputDevice.get(),
                            actual.size() * sizeof(float),
                            hipMemcpyDeviceToHost),
                  "prepared output D2H")) {
      return false;
    }

    std::vector<Q8_1Block> actualQ8(expectedQ8.size());
    if (!checkHip(hipMemcpy(actualQ8.data(), scratchDevice.get(),
                            actualQ8.size() * sizeof(Q8_1Block),
                            hipMemcpyDeviceToHost),
                  "prepared Q8_1 D2H") ||
        std::memcmp(actualQ8.data(), expectedQ8.data(),
                    expectedQ8.size() * sizeof(Q8_1Block)) != 0) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s prepared Q8_1 mismatch\n",
                   name);
      return false;
    }
    for (int64_t row = 0; row < outputElements; ++row) {
      if (!closeEnough(actual[static_cast<size_t>(row)],
                       expected[static_cast<size_t>(row)])) {
        std::fprintf(stderr,
                     "ggml_k_quant_matvec_test: %s prepared row %lld "
                     "mismatch: actual=%g expected=%g\n",
                     name, static_cast<long long>(row),
                     static_cast<double>(actual[static_cast<size_t>(row)]),
                     static_cast<double>(expected[static_cast<size_t>(row)]));
        return false;
      }
    }
    for (size_t index = static_cast<size_t>(outputElements);
         index < actual.size(); ++index) {
      if (actual[index] != kGuard) {
        std::fprintf(stderr,
                     "ggml_k_quant_matvec_test: %s prepared output guard %zu "
                     "was overwritten\n",
                     name, index - static_cast<size_t>(outputElements));
        return false;
      }
    }
  }
  std::printf("%s batch=%lld M=%lld N=%lld K=%lld passed (max_abs_diff=%g)\n",
              name, static_cast<long long>(batchCount),
              static_cast<long long>(M), static_cast<long long>(rows),
              static_cast<long long>(columns),
              static_cast<double>(maximumDifference));
  return true;
}

bool runQ4KCase(int64_t rows, int64_t columns, int64_t M, int64_t batchCount,
                const char *name, bool useQ8) {
  const int64_t blocksPerRow = columns / kQkK;
  std::vector<float> activation = makeActivation(M * batchCount, columns);
  std::vector<Q4KBlock> packed = makeKQuantWeights<Q4KBlock>(
      rows * blocksPerRow, /*seedOffset=*/1);
  std::vector<Q8_1Block> q8 =
      useQ8 ? quantizeQ8_1Reference(
                  activation,
                  M * batchCount == 1
                      ? HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM
                      : HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED)
            : std::vector<Q8_1Block>{};
  std::vector<float> referenceActivation =
      useQ8 ? dequantizeQ8_1Reference(q8) : activation;
  if (useQ8 && !verifyLossyQ8_1(activation, referenceActivation, name)) {
    return false;
  }
  // One row, or two to eight through llama.cpp's MMVQ columns, take the
  // minimum term from the quants' integer sums.
  const bool integerSums =
      M * batchCount == 1 || (M >= 2 && M <= 8);
  std::vector<float> expected(static_cast<size_t>(batchCount * M * rows));
  std::vector<float> denseExpected(expected.size());
  float maximumQ8Delta = 0.0f;
  for (int64_t activationRow = 0; activationRow < batchCount * M;
       ++activationRow) {
    for (int64_t row = 0; row < rows; ++row) {
      const size_t outputIndex =
          static_cast<size_t>(activationRow * rows + row);
      denseExpected[outputIndex] = hip::kquant::referenceDotQ4K(
          packed.data() + row * blocksPerRow, static_cast<size_t>(blocksPerRow),
          activation.data() + activationRow * columns);
      expected[outputIndex] =
          useQ8 ? referenceDotQ8_1(
                      packed.data() + row * blocksPerRow,
                      static_cast<size_t>(blocksPerRow),
                      q8.data() + activationRow *
                                      (columns / hip::kquant::kQ8_1BlockElements),
                      integerSums)
                : denseExpected[outputIndex];
      maximumQ8Delta =
          std::max(maximumQ8Delta, std::fabs(expected[outputIndex] -
                                             denseExpected[outputIndex]));
    }
  }
  if (useQ8) {
    std::printf("%s reference max_abs_vs_dequantized_f32=%g\n", name,
                static_cast<double>(maximumQ8Delta));
  }
  return copyAndRun(packed, activation, expected, rows, columns, M, batchCount,
                    name, HIP_KQUANT_FORMAT_Q4_K, q8);
}

bool runQ6KCase(int64_t rows, int64_t columns, int64_t M, int64_t batchCount,
                const char *name, bool useQ8) {
  const int64_t blocksPerRow = columns / kQkK;
  std::vector<float> activation = makeActivation(M * batchCount, columns);
  std::vector<Q6KBlock> packed = makeKQuantWeights<Q6KBlock>(
      rows * blocksPerRow, /*seedOffset=*/1);
  std::vector<Q8_1Block> q8 =
      useQ8 ? quantizeQ8_1Reference(
                  activation,
                  M * batchCount == 1
                      ? HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM
                      : HIP_Q8_1_SUM_PREQUANTIZATION)
            : std::vector<Q8_1Block>{};
  std::vector<float> referenceActivation =
      useQ8 ? dequantizeQ8_1Reference(q8) : activation;
  if (useQ8 && !verifyLossyQ8_1(activation, referenceActivation, name)) {
    return false;
  }
  std::vector<float> expected(static_cast<size_t>(batchCount * M * rows));
  std::vector<float> denseExpected(expected.size());
  float maximumQ8Delta = 0.0f;
  for (int64_t activationRow = 0; activationRow < batchCount * M;
       ++activationRow) {
    for (int64_t row = 0; row < rows; ++row) {
      const size_t outputIndex =
          static_cast<size_t>(activationRow * rows + row);
      denseExpected[outputIndex] = hip::kquant::referenceDotQ6K(
          packed.data() + row * blocksPerRow, static_cast<size_t>(blocksPerRow),
          activation.data() + activationRow * columns);
      expected[outputIndex] = hip::kquant::referenceDotQ6K(
          packed.data() + row * blocksPerRow, static_cast<size_t>(blocksPerRow),
          referenceActivation.data() + activationRow * columns);
      maximumQ8Delta =
          std::max(maximumQ8Delta, std::fabs(expected[outputIndex] -
                                             denseExpected[outputIndex]));
    }
  }
  if (useQ8) {
    std::printf("%s reference max_abs_vs_dequantized_f32=%g\n", name,
                static_cast<double>(maximumQ8Delta));
  }
  return copyAndRun(packed, activation, expected, rows, columns, M, batchCount,
                    name, HIP_KQUANT_FORMAT_Q6_K, q8);
}

template <typename Block>
bool runMmqCase(int formatId, int64_t N, int64_t K, int64_t M,
                int64_t batchCount, bool quantizeFromF32, const char *name,
                bool zeroActivation = false) {
  if constexpr (std::is_same_v<Block, Q4KBlock>) {
    if (quantizeFromF32) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s requested unsupported Q4_K "
                   "F32 MMQ quantization\n",
                   name);
      return false;
    }
  }
  constexpr size_t guardElements = 7;
  const int64_t blocksPerWeightRow = K / kQkK;
  const int64_t outputElements = batchCount * M * N;
  const std::vector<float> activation =
      zeroActivation
          ? std::vector<float>(static_cast<size_t>(batchCount * M * K), 0.0f)
          : makeActivation(batchCount * M, K);
  const std::vector<Block> packed =
      makeKQuantWeights<Block>(N * blocksPerWeightRow, /*seedOffset=*/1);
  const std::vector<Q8_1Block> q8 = quantizeQ8_1Reference(activation);
  const MmqQ8_1Reference f32Q8Reference =
      quantizeFromF32 ? quantizeMmqQ8_1Reference(activation, M, K, batchCount)
                      : MmqQ8_1Reference{};
  const std::vector<float> referenceActivation =
      quantizeFromF32 ? f32Q8Reference.dequantized
                      : dequantizeQ8_1Reference(q8);
  std::vector<float> expected(static_cast<size_t>(outputElements));
  bool q4SumConventionObserved = false;
  for (int64_t activationRow = 0; activationRow < batchCount * M;
       ++activationRow) {
    for (int64_t outputRow = 0; outputRow < N; ++outputRow) {
      const Block *weight = packed.data() + outputRow * blocksPerWeightRow;
      const float *input = referenceActivation.data() + activationRow * K;
      if constexpr (std::is_same_v<Block, Q4KBlock>) {
        const size_t outputIndex =
            static_cast<size_t>(activationRow * N + outputRow);
        expected[outputIndex] = referenceDotQ8_1(
            weight, static_cast<size_t>(blocksPerWeightRow),
            q8.data() + activationRow * (K / hip::kquant::kQ8_1BlockElements));
        const float reconstructed = hip::kquant::referenceDotQ4K(
            weight, static_cast<size_t>(blocksPerWeightRow), input);
        q4SumConventionObserved |=
            std::fabs(expected[outputIndex] - reconstructed) > 1.0e-3f;
      } else {
        expected[static_cast<size_t>(activationRow * N + outputRow)] =
            hip::kquant::referenceDotQ6K(
                weight, static_cast<size_t>(blocksPerWeightRow), input);
      }
    }
  }
  if constexpr (std::is_same_v<Block, Q4KBlock>) {
    if (!zeroActivation && !q4SumConventionObserved) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s did not distinguish the "
                   "stored Q8_1 sum convention\n",
                   name);
      return false;
    }
  }

  const size_t q8Bytes = q8.size() * sizeof(Q8_1Block);
  const size_t mmqBytes =
      q8Bytes + HIP_KQUANT_MMQ_INPUT_TILE *
                    HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE *
                    static_cast<size_t>(sizeof(Q8_1Block));
  std::vector<float> actual(static_cast<size_t>(outputElements) + guardElements,
                            kGuard);
  DeviceBuffer activationDevice(activation.size() * sizeof(float));
  DeviceBuffer q8Device(mmqBytes);
  DeviceBuffer mmqDevice(mmqBytes);
  DeviceBuffer weightsDevice(packed.size() * sizeof(Block));
  DeviceBuffer outputDevice(actual.size() * sizeof(float));
  if (!activationDevice.get() || !q8Device.get() || !mmqDevice.get() ||
      !weightsDevice.get() || !outputDevice.get() ||
      !checkHip(hipMemcpy(activationDevice.get(), activation.data(),
                          activation.size() * sizeof(float),
                          hipMemcpyHostToDevice),
                "MMQ activation H2D") ||
      !checkHip(
          hipMemcpy(q8Device.get(), q8.data(), q8Bytes, hipMemcpyHostToDevice),
          "MMQ Q8_1 H2D") ||
      !checkHip(hipMemcpy(weightsDevice.get(), packed.data(),
                          packed.size() * sizeof(Block), hipMemcpyHostToDevice),
                "MMQ weights H2D") ||
      !checkHip(hipMemcpy(outputDevice.get(), actual.data(),
                          actual.size() * sizeof(float), hipMemcpyHostToDevice),
                "MMQ output H2D")) {
    return false;
  }

  const int status =
      quantizeFromF32
          ? hip_kquant_mmq_f32(
                nullptr, static_cast<const float *>(activationDevice.get()),
                q8Device.get(), weightsDevice.get(),
                static_cast<float *>(outputDevice.get()), M, N, K, batchCount,
                formatId)
          : hip_kquant_mmq_from_q8_1(
                nullptr, q8Device.get(), mmqDevice.get(), weightsDevice.get(),
                static_cast<float *>(outputDevice.get()), M, N, K, batchCount,
                formatId);
  if (status != 0 ||
      !checkHip(hipMemcpy(actual.data(), outputDevice.get(),
                          actual.size() * sizeof(float), hipMemcpyDeviceToHost),
                "MMQ output D2H")) {
    return false;
  }

  {
    const std::vector<uint8_t> expectedMmq =
        quantizeFromF32
            ? f32Q8Reference.packed
            : packMmqQ8_1Reference(q8, M, K, batchCount,
                                   /*storeSums=*/formatId ==
                                       HIP_KQUANT_FORMAT_Q4_K);
    std::vector<uint8_t> actualMmq(expectedMmq.size());
    const void *actualMmqDevice =
        quantizeFromF32 ? q8Device.get() : mmqDevice.get();
    if (!checkHip(hipMemcpy(actualMmq.data(), actualMmqDevice, actualMmq.size(),
                            hipMemcpyDeviceToHost),
                  "MMQ packed Q8_1 D2H")) {
      return false;
    }
    if (actualMmq != expectedMmq) {
      size_t first = 0;
      while (first < actualMmq.size() && actualMmq[first] == expectedMmq[first])
        ++first;
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s MMQ activation layout "
                   "mismatch at byte %zu: actual=%u expected=%u\n",
                   name, first, static_cast<unsigned>(actualMmq[first]),
                   static_cast<unsigned>(expectedMmq[first]));
      return false;
    }
  }

  float maximumDifference = 0.0f;
  // Q4_K consumes stored FP16 block sums, while Q6_K uses either the direct
  // D4 quantizer or standard Q8_1 blocks. Keep their observed reassociation
  // allowances separate so the Q4_K cancellation bound does not weaken Q6_K.
  const float absoluteTolerance = formatId == HIP_KQUANT_FORMAT_Q4_K
                                      ? 1.0f
                                      : (quantizeFromF32 ? 0.04f : 0.03f);
  const float relativeTolerance = formatId == HIP_KQUANT_FORMAT_Q4_K
                                      ? 3.0e-3f
                                      : (quantizeFromF32 ? 1.0e-3f : 5.0e-4f);
  double absoluteDifferenceSum = 0.0;
  double dotProduct = 0.0;
  double actualNorm = 0.0;
  double expectedNorm = 0.0;
  for (int64_t index = 0; index < outputElements; ++index) {
    const float actualValue = actual[static_cast<size_t>(index)];
    const float expectedValue = expected[static_cast<size_t>(index)];
    const float difference = std::fabs(actualValue - expectedValue);
    maximumDifference = std::max(maximumDifference, difference);
    if (!std::isfinite(actualValue)) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s output %lld is non-finite\n",
                   name, static_cast<long long>(index));
      return false;
    }
    if (zeroActivation && actualValue != 0.0f) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s zero input produced %g at "
                   "output %lld\n",
                   name, static_cast<double>(actualValue),
                   static_cast<long long>(index));
      return false;
    }
    // The pinned integer-WMMA kernel groups products differently from the
    // scalar host oracle. Its Q4_K path also rounds the effective scale and
    // minimum coefficients to F16 before accumulation. Bound those declared
    // differences per path; the full model has a separate exact-token gate.
    if (difference >
        absoluteTolerance + relativeTolerance * std::fabs(expectedValue)) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s output %lld mismatch: "
                   "actual=%g expected=%g\n",
                   name, static_cast<long long>(index),
                   static_cast<double>(actualValue),
                   static_cast<double>(expectedValue));
      return false;
    }
    absoluteDifferenceSum += difference;
    dotProduct += static_cast<double>(actualValue) * expectedValue;
    actualNorm += static_cast<double>(actualValue) * actualValue;
    expectedNorm += static_cast<double>(expectedValue) * expectedValue;
  }
  for (size_t index = static_cast<size_t>(outputElements);
       index < actual.size(); ++index) {
    if (actual[index] != kGuard) {
      std::fprintf(stderr,
                   "ggml_k_quant_matvec_test: %s wrote output guard %zu\n",
                   name, index - static_cast<size_t>(outputElements));
      return false;
    }
  }
  const double meanAbsoluteDifference =
      absoluteDifferenceSum / static_cast<double>(outputElements);
  const double cosineSimilarity =
      actualNorm == 0.0 && expectedNorm == 0.0
          ? 1.0
          : dotProduct / std::sqrt(actualNorm * expectedNorm);
  if (!std::isfinite(cosineSimilarity) || cosineSimilarity < 0.99999) {
    std::fprintf(stderr,
                 "ggml_k_quant_matvec_test: %s exceeded MMQ tolerance: "
                 "max_abs=%g mean_abs=%g cosine=%g\n",
                 name, static_cast<double>(maximumDifference),
                 meanAbsoluteDifference, cosineSimilarity);
    return false;
  }
  std::printf("%s batch=%lld M=%lld N=%lld K=%lld passed "
              "(max_abs_diff=%g mean_abs_diff=%g cosine=%g)\n",
              name, static_cast<long long>(batchCount),
              static_cast<long long>(M), static_cast<long long>(N),
              static_cast<long long>(K), static_cast<double>(maximumDifference),
              meanAbsoluteDifference, cosineSimilarity);
  return true;
}

} // namespace

int main() {
  const int device = hipdnn_ep::test::findGfx1151Device(kTestName);
  if (device == -1) {
    std::fprintf(stderr,
                 "ggml_k_quant_matvec_test: no gfx1151 device; skipping\n");
    return 77;
  }
  if (device < 0 || !checkHip(hipSetDevice(device), "hipSetDevice")) {
    return 1;
  }
  const bool q4Gemv = runQ4KCase(/*N=*/77, /*K=*/1024, /*M=*/1,
                                 /*batchCount=*/3, "Q4_K Q8_1 MMVQ", true);
  const bool q6Gemv = runQ6KCase(/*N=*/53, /*K=*/1280, /*M=*/1,
                                 /*batchCount=*/2, "Q6_K Q8_1 MMVQ", true);
  constexpr std::array<int64_t, 9> rowsToTest = {2, 3, 4, 5, 8, 9, 16, 17, 40};
  bool automatic = true;
  for (int64_t M : rowsToTest) {
    const std::string q4Name = "Q4_K Q8_1 auto M" + std::to_string(M);
    const std::string q6Name = "Q6_K Q8_1 auto M" + std::to_string(M);
    automatic &= runQ4KCase(/*N=*/7, /*K=*/4096, M, /*batchCount=*/2,
                            q4Name.c_str(), true);
    automatic &= runQ6KCase(/*N=*/5, /*K=*/14336, M, /*batchCount=*/2,
                            q6Name.c_str(), true);
  }
  const bool q4Mmq = runMmqCase<Q4KBlock>(
      HIP_KQUANT_FORMAT_Q4_K, /*N=*/256, /*K=*/4096, /*M=*/40,
      /*batchCount=*/2, /*quantizeFromF32=*/false, "Q4_K MMQ prefill");
  const bool q6Mmq = runMmqCase<Q6KBlock>(
      HIP_KQUANT_FORMAT_Q6_K, /*N=*/128, /*K=*/14336, /*M=*/40,
      /*batchCount=*/1, /*quantizeFromF32=*/true, "Q6_K MMQ prefill");
  return q4Gemv && q6Gemv && automatic && q4Mmq && q6Mmq ? 0 : 1;
}
