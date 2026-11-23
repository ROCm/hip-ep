/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipTestSupport.h"
#include "hip_custom_kernels.h"
#include "kquant_abi.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr char kTestName[] = "skip_rms_norm_q8_1_test";
using hipdnn_ep::test::DeviceBuffer;

constexpr int kThreads = 256;
constexpr int64_t kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-5f;
constexpr size_t kGuardBytes = 64;
constexpr uint8_t kGuardValue = 0xa5;

using Q8_1Block = hip_q8_1_block_t;

float halfBitsToFloat(uint16_t bits) {
  _Float16 value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

bool checkHip(hipError_t error, const char *operation) {
  return hipdnn_ep::test::checkHip(error, operation, kTestName);
}

// This staged reference deliberately uses the same F32 per-thread accumulation
// and block-reduction order as the fused kernel. It materializes normalized,
// then calls the independently tested public Q8_1 preparation entry point.
// Therefore byte equality below proves that fusion did not change the Q8_1
// rounding contract by removing the normalized intermediate.
__global__ void referenceAddRmsNorm(const float *__restrict__ lhs,
                                    const float *__restrict__ rhs,
                                    const float *__restrict__ scale,
                                    float *__restrict__ residual,
                                    float *__restrict__ normalized, int64_t K,
                                    float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t rowOffset = row * K;
  __shared__ float reduction[kThreads];
  __shared__ float inverseRms;

  float localSumSquares = 0.0f;
  for (int64_t column = threadIdx.x; column < K; column += blockDim.x) {
    const int64_t index = rowOffset + column;
    const float value = lhs[index] + rhs[index];
    residual[index] = value;
    localSumSquares += value * value;
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
  for (int64_t column = threadIdx.x; column < K; column += blockDim.x) {
    const int64_t index = rowOffset + column;
    normalized[index] = residual[index] * inverseRms * scale[column];
  }
}

bool hasGuard(const std::vector<uint8_t> &bytes, size_t payloadBytes,
              const char *name) {
  for (size_t index = payloadBytes; index < bytes.size(); ++index) {
    if (bytes[index] != kGuardValue) {
      std::fprintf(stderr,
                   "ggml_add_rms_norm_q8_test: %s guard byte %zu changed\n",
                   name, index - payloadBytes);
      return false;
    }
  }
  return true;
}

bool verifyCanonicalSemantics(const std::vector<float> &lhs,
                              const std::vector<float> &rhs,
                              const std::vector<float> &scale,
                              const std::vector<uint8_t> &residualBytes,
                              const std::vector<uint8_t> &q8Bytes, int64_t rows,
                              int sumMode, int64_t rowsPerToken,
                              const char *name) {
  constexpr float kNormalizedAbsoluteTolerance = 2.5e-2f;
  constexpr float kSumAbsoluteTolerance = 5.0e-2f;
  const int64_t blocksPerRow = kHiddenSize / HIP_Q8_1_BLOCK_ELEMENTS;
  for (int64_t row = 0; row < rows; ++row) {
    double sumSquares = 0.0;
    for (int64_t column = 0; column < kHiddenSize; ++column) {
      const size_t index = static_cast<size_t>(row * kHiddenSize + column);
      const float expectedResidual = lhs[index] + rhs[index];
      float actualResidual;
      std::memcpy(&actualResidual, residualBytes.data() + index * sizeof(float),
                  sizeof(actualResidual));
      if (actualResidual != expectedResidual) {
        std::fprintf(stderr,
                     "ggml_add_rms_norm_q8_test: %s canonical residual "
                     "mismatch at %zu\n",
                     name, index);
        return false;
      }
      sumSquares += static_cast<double>(expectedResidual) * expectedResidual;
    }
    const double inverseRms =
        1.0 / std::sqrt(sumSquares / static_cast<double>(kHiddenSize) +
                        static_cast<double>(kEpsilon));
    for (int64_t blockIndex = 0; blockIndex < blocksPerRow; ++blockIndex) {
      Q8_1Block block{};
      const size_t byteOffset =
          static_cast<size_t>(row * blocksPerRow + blockIndex) * sizeof(block);
      std::memcpy(&block, q8Bytes.data() + byteOffset, sizeof(block));
      const float q8Scale = halfBitsToFloat(block.d);
      double expectedBlockSum = 0.0;
      for (int64_t lane = 0; lane < HIP_Q8_1_BLOCK_ELEMENTS; ++lane) {
        const size_t index = static_cast<size_t>(
            row * kHiddenSize + blockIndex * HIP_Q8_1_BLOCK_ELEMENTS +
            lane);
        const float residual = lhs[index] + rhs[index];
        const float expectedNormalized =
            static_cast<float>(static_cast<double>(residual) * inverseRms *
                               scale[index % static_cast<size_t>(kHiddenSize)]);
        const float dequantized = q8Scale * static_cast<float>(block.qs[lane]);
        if (std::fabs(dequantized - expectedNormalized) >
            kNormalizedAbsoluteTolerance) {
          std::fprintf(stderr,
                       "ggml_add_rms_norm_q8_test: %s canonical Q8 mismatch "
                       "at %zu: dequantized=%g expected=%g\n",
                       name, index, static_cast<double>(dequantized),
                       static_cast<double>(expectedNormalized));
          return false;
        }
        expectedBlockSum += expectedNormalized;
      }
      const float storedSum = halfBitsToFloat(block.s);
      if (sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM &&
          rows / rowsPerToken == 1) {
        int16_t actualIntegerSum;
        std::memcpy(&actualIntegerSum, &block.s, sizeof(actualIntegerSum));
        int expectedIntegerSum = 0;
        for (int8_t quant : block.qs)
          expectedIntegerSum += quant;
        if (actualIntegerSum != expectedIntegerSum) {
          std::fprintf(stderr,
                       "ggml_add_rms_norm_q8_test: %s Q8 integer sum "
                       "mismatch at row %lld block %lld: actual=%d "
                       "expected=%d\n",
                       name, static_cast<long long>(row),
                       static_cast<long long>(blockIndex), actualIntegerSum,
                       expectedIntegerSum);
          return false;
        }
        continue;
      }
      if (sumMode == HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED ||
          sumMode == HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM) {
        int integerSum = 0;
        for (int8_t quant : block.qs)
          integerSum += quant;
        expectedBlockSum = static_cast<double>(q8Scale) * integerSum;
      }
      const float sumTolerance =
          kSumAbsoluteTolerance +
          2.0e-3f * std::fabs(static_cast<float>(expectedBlockSum));
      if (std::fabs(storedSum - static_cast<float>(expectedBlockSum)) >
          sumTolerance) {
        std::fprintf(stderr,
                     "ggml_add_rms_norm_q8_test: %s Q8 sum mismatch at row "
                     "%lld block %lld: actual=%g expected=%g\n",
                     name, static_cast<long long>(row),
                     static_cast<long long>(blockIndex),
                     static_cast<double>(storedSum), expectedBlockSum);
        return false;
      }
    }
  }
  return true;
}

bool runCase(hipStream_t stream, int64_t rows, int sumMode,
             int64_t rowsPerToken, const char *name) {
  const size_t elementCount = static_cast<size_t>(rows * kHiddenSize);
  const size_t tensorBytes = elementCount * sizeof(float);
  const size_t q8Bytes =
      static_cast<size_t>(rows * (kHiddenSize / HIP_Q8_1_BLOCK_ELEMENTS) *
                          HIP_Q8_1_BLOCK_SIZE);

  std::vector<float> lhs(elementCount);
  std::vector<float> rhs(elementCount);
  std::vector<float> scale(static_cast<size_t>(kHiddenSize));
  for (size_t index = 0; index < elementCount; ++index) {
    lhs[index] = 0.75f * std::sin(static_cast<float>(index + 3) * 0.013f);
    rhs[index] = 0.5f * std::cos(static_cast<float>(index + 11) * 0.017f);
  }
  for (size_t index = 0; index < scale.size(); ++index)
    scale[index] = 0.7f + 0.0001f * static_cast<float>(index % 997);

  DeviceBuffer deviceLhs(tensorBytes);
  DeviceBuffer deviceRhs(tensorBytes);
  DeviceBuffer deviceScale(scale.size() * sizeof(float));
  DeviceBuffer referenceResidual(tensorBytes);
  DeviceBuffer referenceNormalized(tensorBytes);
  DeviceBuffer referenceQ8(q8Bytes + kGuardBytes);
  DeviceBuffer fusedResidual(tensorBytes + kGuardBytes);
  DeviceBuffer fusedQ8(q8Bytes + kGuardBytes);
  if (!deviceLhs.get() || !deviceRhs.get() || !deviceScale.get() ||
      !referenceResidual.get() || !referenceNormalized.get() ||
      !referenceQ8.get() || !fusedResidual.get() || !fusedQ8.get()) {
    std::fprintf(stderr, "ggml_add_rms_norm_q8_test: %s allocation failed\n",
                 name);
    return false;
  }

  std::vector<uint8_t> residualInit(tensorBytes + kGuardBytes, kGuardValue);
  std::vector<uint8_t> q8Init(q8Bytes + kGuardBytes, kGuardValue);
  if (!checkHip(hipMemcpyAsync(deviceLhs.get(), lhs.data(), tensorBytes,
                               hipMemcpyHostToDevice, stream),
                "lhs H2D") ||
      !checkHip(hipMemcpyAsync(deviceRhs.get(), rhs.data(), tensorBytes,
                               hipMemcpyHostToDevice, stream),
                "rhs H2D") ||
      !checkHip(hipMemcpyAsync(deviceScale.get(), scale.data(),
                               scale.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "scale H2D") ||
      !checkHip(hipMemcpyAsync(fusedResidual.get(), residualInit.data(),
                               residualInit.size(), hipMemcpyHostToDevice,
                               stream),
                "fused residual init H2D") ||
      !checkHip(hipMemcpyAsync(referenceQ8.get(), q8Init.data(), q8Init.size(),
                               hipMemcpyHostToDevice, stream),
                "reference Q8 init H2D") ||
      !checkHip(hipMemcpyAsync(fusedQ8.get(), q8Init.data(), q8Init.size(),
                               hipMemcpyHostToDevice, stream),
                "fused Q8 init H2D")) {
    return false;
  }

  hipLaunchKernelGGL(
      referenceAddRmsNorm, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
      stream, static_cast<const float *>(deviceLhs.get()),
      static_cast<const float *>(deviceRhs.get()),
      static_cast<const float *>(deviceScale.get()),
      static_cast<float *>(referenceResidual.get()),
      static_cast<float *>(referenceNormalized.get()), kHiddenSize, kEpsilon);
  if (!checkHip(hipGetLastError(), "reference add/rms launch") ||
      hip_quantize_q8_1(
          stream, static_cast<const float *>(referenceNormalized.get()),
          referenceQ8.get(), rows, kHiddenSize, /*batch_count=*/1, sumMode,
          rowsPerToken) != 0 ||
      hip_skip_rms_norm_q8_1(
          stream, static_cast<const float *>(deviceLhs.get()),
          static_cast<const float *>(deviceRhs.get()),
          static_cast<const float *>(deviceScale.get()),
          static_cast<float *>(fusedResidual.get()), fusedQ8.get(), rows,
          kHiddenSize, kEpsilon, sumMode, rowsPerToken) != 0) {
    std::fprintf(stderr, "ggml_add_rms_norm_q8_test: %s kernel launch failed\n",
                 name);
    return false;
  }

  std::vector<float> expectedResidual(elementCount);
  std::vector<uint8_t> actualResidual(tensorBytes + kGuardBytes);
  std::vector<uint8_t> expectedQ8(q8Bytes + kGuardBytes);
  std::vector<uint8_t> actualQ8(q8Bytes + kGuardBytes);
  if (!checkHip(hipMemcpyAsync(expectedResidual.data(), referenceResidual.get(),
                               tensorBytes, hipMemcpyDeviceToHost, stream),
                "reference residual D2H") ||
      !checkHip(hipMemcpyAsync(actualResidual.data(), fusedResidual.get(),
                               actualResidual.size(), hipMemcpyDeviceToHost,
                               stream),
                "fused residual D2H") ||
      !checkHip(hipMemcpyAsync(expectedQ8.data(), referenceQ8.get(),
                               expectedQ8.size(), hipMemcpyDeviceToHost,
                               stream),
                "reference Q8 D2H") ||
      !checkHip(hipMemcpyAsync(actualQ8.data(), fusedQ8.get(), actualQ8.size(),
                               hipMemcpyDeviceToHost, stream),
                "fused Q8 D2H") ||
      !checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize")) {
    return false;
  }

  if (std::memcmp(actualResidual.data(), expectedResidual.data(),
                  tensorBytes) != 0) {
    std::fprintf(stderr,
                 "ggml_add_rms_norm_q8_test: %s residual bytes differ\n", name);
    return false;
  }
  if (std::memcmp(actualQ8.data(), expectedQ8.data(), q8Bytes) != 0) {
    size_t first = 0;
    while (first < q8Bytes && actualQ8[first] == expectedQ8[first])
      ++first;
    std::fprintf(stderr,
                 "ggml_add_rms_norm_q8_test: %s Q8_1 bytes differ at %zu "
                 "(actual=%u expected=%u)\n",
                 name, first, static_cast<unsigned>(actualQ8[first]),
                 static_cast<unsigned>(expectedQ8[first]));
    return false;
  }
  if (!hasGuard(actualResidual, tensorBytes, "residual") ||
      !hasGuard(actualQ8, q8Bytes, "fused Q8") ||
      !hasGuard(expectedQ8, q8Bytes, "reference Q8") ||
      !verifyCanonicalSemantics(lhs, rhs, scale, actualResidual, actualQ8, rows,
                                sumMode, rowsPerToken, name)) {
    return false;
  }

  std::printf("%s rows=%lld K=%lld passed with byte-identical residual and "
              "Q8_1 output\n",
              name, static_cast<long long>(rows),
              static_cast<long long>(kHiddenSize));
  return true;
}

} // namespace

int main() {
  const int device = hipdnn_ep::test::findGfx1151Device(kTestName);
  if (device == -1) {
    std::fprintf(stderr,
                 "ggml_add_rms_norm_q8_test: no gfx1151 device; skipping\n");
    return 77;
  }
  if (device < 0 || !checkHip(hipSetDevice(device), "hipSetDevice"))
    return 1;

  hipStream_t stream = nullptr;
  if (!checkHip(hipStreamCreate(&stream), "hipStreamCreate"))
    return 1;

  const bool legacy = runCase(stream, 1, HIP_Q8_1_SUM_PREQUANTIZATION,
                              /*rowsPerToken=*/1, "legacy decode");
  const bool decode =
      runCase(stream, 1, HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM,
              /*rowsPerToken=*/1, "integer-sum decode");
  const bool routedDecode =
      runCase(stream, 8, HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM,
              /*rowsPerToken=*/8, "integer-sum routed decode");
  const bool prefill =
      runCase(stream, 40, HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM,
              /*rowsPerToken=*/1, "scaled-sum prefill");
  const bool rejectsNull =
      hip_skip_rms_norm_q8_1(stream, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, 1, kHiddenSize, kEpsilon,
                                 HIP_Q8_1_SUM_PREQUANTIZATION,
                                 /*rows_per_token=*/1) != 0;
  const bool rejectsK =
      hip_skip_rms_norm_q8_1(stream, reinterpret_cast<const float *>(1),
                                 reinterpret_cast<const float *>(1),
                                 reinterpret_cast<const float *>(1),
                                 reinterpret_cast<float *>(1),
                                 reinterpret_cast<void *>(4), 1, 128, kEpsilon,
                                 HIP_Q8_1_SUM_PREQUANTIZATION,
                                 /*rows_per_token=*/1) != 0;
  const bool rejectsEpsilon =
      hip_skip_rms_norm_q8_1(stream, reinterpret_cast<const float *>(1),
                                 reinterpret_cast<const float *>(1),
                                 reinterpret_cast<const float *>(1),
                                 reinterpret_cast<float *>(1),
                                 reinterpret_cast<void *>(4), 1, kHiddenSize,
                                 -1.0f, HIP_Q8_1_SUM_PREQUANTIZATION,
                                 /*rows_per_token=*/1) != 0;

  const bool destroyed = checkHip(hipStreamDestroy(stream), "stream destroy");
  return legacy && decode && routedDecode && prefill && rejectsNull &&
                 rejectsK && rejectsEpsilon && destroyed
             ? 0
             : 1;
}
