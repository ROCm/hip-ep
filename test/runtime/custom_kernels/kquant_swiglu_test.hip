/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "KQuantTestSupport.h"
#include "HipTestSupport.h"
#include "hip_custom_kernels.h"
#include "kquant_abi.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

using hip::kquant::Q4KBlock;
using hip::kquant::Q6KBlock;
using hipdnn_ep::test::DeviceBuffer;
using hipdnn_ep::test::kquant::makeKQuantWeights;
using hipdnn_ep::test::kquant::Q8_1Block;
using hipdnn_ep::test::kquant::referenceDotQ8_1;
constexpr int64_t kK = 4096;
constexpr float kGuard = -9876.5f;
constexpr char kTestName[] = "kquant_swiglu_test";

static_assert(sizeof(Q8_1Block) == HIP_Q8_1_BLOCK_SIZE);

bool checkHip(hipError_t error, const char *operation) {
  return hipdnn_ep::test::checkHip(error, operation, kTestName);
}

template <typename GateBlock, typename UpBlock>
bool runCase(hipStream_t stream, int64_t M, int64_t N, int64_t batchCount,
             int gateFormat, int upFormat, const char *name) {
  const int64_t weightBlocks = N * (kK / hip::kquant::kKQuantBlockElements);
  constexpr uint32_t gateSeedBias =
      std::is_same_v<GateBlock, Q4KBlock> ? 1U : 7U;
  constexpr uint32_t upSeedBias = std::is_same_v<UpBlock, Q4KBlock> ? 1U : 7U;
  std::vector<GateBlock> gateWeights = makeKQuantWeights<GateBlock>(
      weightBlocks, /*seedOffset=*/3 + gateSeedBias);
  std::vector<UpBlock> upWeights =
      makeKQuantWeights<UpBlock>(weightBlocks, /*seedOffset=*/101 + upSeedBias);
  const size_t activationElements = static_cast<size_t>(batchCount * M * kK);
  const size_t outputElements = static_cast<size_t>(batchCount * M * N);
  const size_t q8Bytes = activationElements / HIP_Q8_1_BLOCK_ELEMENTS *
                         HIP_Q8_1_BLOCK_SIZE;
  std::vector<float> activation(activationElements);
  for (size_t index = 0; index < activation.size(); ++index) {
    activation[index] =
        0.8f * std::cos(static_cast<float>(index + 5) * 0.019f) +
        static_cast<float>(static_cast<int>(index % 31) - 15) * 0.017f;
  }
  constexpr size_t guardElements = 16;
  std::vector<float> expected(outputElements + guardElements, kGuard);
  std::vector<float> actual(outputElements + guardElements, kGuard);
  std::vector<float> actualGate(outputElements);
  std::vector<float> actualUp(outputElements);
  std::vector<float> scratch(
      (M > 1 ? outputElements : size_t{1}) + guardElements, kGuard);
  std::vector<Q8_1Block> hostQ8(q8Bytes / sizeof(Q8_1Block));

  DeviceBuffer activationDevice(activation.size() * sizeof(float));
  DeviceBuffer q8Device(q8Bytes);
  DeviceBuffer gateWeightDevice(gateWeights.size() * sizeof(GateBlock));
  DeviceBuffer upWeightDevice(upWeights.size() * sizeof(UpBlock));
  DeviceBuffer gateOutputDevice(outputElements * sizeof(float));
  DeviceBuffer upOutputDevice(outputElements * sizeof(float));
  DeviceBuffer expectedDevice(expected.size() * sizeof(float));
  DeviceBuffer actualDevice(actual.size() * sizeof(float));
  DeviceBuffer scratchDevice(scratch.size() * sizeof(float));
  if (!activationDevice.get() || !q8Device.get() || !gateWeightDevice.get() ||
      !upWeightDevice.get() || !gateOutputDevice.get() ||
      !upOutputDevice.get() || !expectedDevice.get() || !actualDevice.get() ||
      !scratchDevice.get()) {
    std::fprintf(stderr, "ggml_k_quant_swiglu_test: %s allocation failed\n",
                 name);
    return false;
  }

  if (!checkHip(hipMemcpyAsync(activationDevice.get(), activation.data(),
                               activation.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "activation H2D") ||
      !checkHip(hipMemcpyAsync(gateWeightDevice.get(), gateWeights.data(),
                               gateWeights.size() * sizeof(GateBlock),
                               hipMemcpyHostToDevice, stream),
                "gate weights H2D") ||
      !checkHip(hipMemcpyAsync(upWeightDevice.get(), upWeights.data(),
                               upWeights.size() * sizeof(UpBlock),
                               hipMemcpyHostToDevice, stream),
                "up weights H2D") ||
      !checkHip(hipMemcpyAsync(expectedDevice.get(), expected.data(),
                               expected.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "expected init H2D") ||
      !checkHip(hipMemcpyAsync(actualDevice.get(), actual.data(),
                               actual.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "actual init H2D") ||
      !checkHip(hipMemcpyAsync(scratchDevice.get(), scratch.data(),
                               scratch.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream),
                "scratch init H2D"))
    return false;

  if (M > 1 &&
      (hip_kquant_swiglu_q8_1(
           stream, q8Device.get(), gateWeightDevice.get(), upWeightDevice.get(),
           static_cast<float *>(actualDevice.get()), M, N, kK, batchCount,
           gateFormat, upFormat, HIP_Q8_1_SUM_PREQUANTIZATION,
           /*rows_per_token=*/1,
           /*output_scratch=*/nullptr, /*output_scratch_size=*/0) == 0 ||
       hip_kquant_swiglu_q8_1(
           stream, q8Device.get(), gateWeightDevice.get(), upWeightDevice.get(),
           static_cast<float *>(actualDevice.get()), M, N, kK, batchCount,
           gateFormat, upFormat, HIP_Q8_1_SUM_PREQUANTIZATION,
           /*rows_per_token=*/1, scratchDevice.get(),
           outputElements * sizeof(float) - 1) == 0 ||
       hip_kquant_swiglu_q8_1(
           stream, q8Device.get(), gateWeightDevice.get(), upWeightDevice.get(),
           static_cast<float *>(actualDevice.get()), M, N, kK, batchCount,
           gateFormat, upFormat, HIP_Q8_1_SUM_PREQUANTIZATION,
           /*rows_per_token=*/1,
           static_cast<uint8_t *>(scratchDevice.get()) + 1,
           outputElements * sizeof(float)) == 0)) {
    std::fprintf(stderr,
                 "ggml_k_quant_swiglu_test: %s accepted invalid scratch\n",
                 name);
    return false;
  }

  if (hip_quantize_q8_1(
          stream, static_cast<const float *>(activationDevice.get()),
          q8Device.get(), M, kK, batchCount, HIP_Q8_1_SUM_PREQUANTIZATION,
          /*rows_per_token=*/1) != 0 ||
      hip_kquant_matmul_q8_1(
          stream, q8Device.get(), gateWeightDevice.get(),
          static_cast<float *>(gateOutputDevice.get()), M, N, kK, batchCount,
          gateFormat, HIP_Q8_1_SUM_PREQUANTIZATION,
          /*rows_per_token=*/1) != 0 ||
      hip_kquant_matmul_q8_1(stream, q8Device.get(), upWeightDevice.get(),
                                   static_cast<float *>(upOutputDevice.get()),
                                   M, N, kK, batchCount, upFormat,
                                   HIP_Q8_1_SUM_PREQUANTIZATION,
                                   /*rows_per_token=*/1) != 0 ||
      hip_swiglu(stream, gateOutputDevice.get(), upOutputDevice.get(),
                 expectedDevice.get(), static_cast<int64_t>(outputElements),
                 HIP_DTYPE_FLOAT32) != 0 ||
      hip_kquant_swiglu_q8_1(
          stream, q8Device.get(), gateWeightDevice.get(), upWeightDevice.get(),
          static_cast<float *>(actualDevice.get()), M, N, kK, batchCount,
          gateFormat, upFormat, HIP_Q8_1_SUM_PREQUANTIZATION,
          /*rows_per_token=*/1,
          M > 1 ? scratchDevice.get() : nullptr,
          M > 1 ? outputElements * sizeof(float) : 0) != 0) {
    std::fprintf(stderr, "ggml_k_quant_swiglu_test: %s launch failed\n", name);
    return false;
  }

  if (!checkHip(hipMemcpyAsync(expected.data(), expectedDevice.get(),
                               expected.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "expected D2H") ||
      !checkHip(hipMemcpyAsync(actual.data(), actualDevice.get(),
                               actual.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "actual D2H") ||
      !checkHip(hipMemcpyAsync(actualGate.data(), gateOutputDevice.get(),
                               actualGate.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "gate D2H") ||
      !checkHip(hipMemcpyAsync(actualUp.data(), upOutputDevice.get(),
                               actualUp.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "up D2H") ||
      !checkHip(hipMemcpyAsync(hostQ8.data(), q8Device.get(), q8Bytes,
                               hipMemcpyDeviceToHost, stream),
                "Q8 D2H") ||
      !checkHip(hipMemcpyAsync(scratch.data(), scratchDevice.get(),
                               scratch.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream),
                "scratch D2H") ||
      !checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize"))
    return false;

  if (std::memcmp(actual.data(), expected.data(),
                  outputElements * sizeof(float)) != 0) {
    std::fprintf(stderr,
                 "ggml_k_quant_swiglu_test: %s differs from the unfused path\n",
                 name);
    return false;
  }
  for (size_t index = outputElements; index < actual.size(); ++index) {
    if (actual[index] != kGuard || expected[index] != kGuard) {
      std::fprintf(stderr,
                   "ggml_k_quant_swiglu_test: %s output guard changed\n", name);
      return false;
    }
  }
  if (M > 1) {
    for (size_t index = outputElements; index < scratch.size(); ++index) {
      if (scratch[index] != kGuard) {
        std::fprintf(stderr,
                     "ggml_k_quant_swiglu_test: %s scratch guard changed\n",
                     name);
        return false;
      }
    }
  }

  // Independent scalar oracle over the packed weights and prepared Q8_1
  // bytes. It does not share the GPU Dot helpers or task indexing.
  const size_t blocksPerActivationRow =
      static_cast<size_t>(kK / hip::kquant::kQ8_1BlockElements);
  const size_t blocksPerWeightRow =
      static_cast<size_t>(kK / hip::kquant::kKQuantBlockElements);
  // Two to eight rows project through llama.cpp's MMVQ columns, which sum
  // the quants themselves.
  const bool integerSums = M >= 2 && M <= 8;
  for (int64_t row = 0; row < batchCount * M; ++row) {
    const Q8_1Block *activationRow =
        hostQ8.data() + static_cast<size_t>(row) * blocksPerActivationRow;
    for (int64_t column = 0; column < N; ++column) {
      const float gate = referenceDotQ8_1(
          gateWeights.data() + static_cast<size_t>(column) * blocksPerWeightRow,
          blocksPerWeightRow, activationRow, integerSums);
      const float up = referenceDotQ8_1(
          upWeights.data() + static_cast<size_t>(column) * blocksPerWeightRow,
          blocksPerWeightRow, activationRow, integerSums);
      const size_t index = static_cast<size_t>(row * N + column);
      const float gateTolerance = 5.0e-2f + 2.0e-4f * std::fabs(gate);
      const float upTolerance = 5.0e-2f + 2.0e-4f * std::fabs(up);
      if (!std::isfinite(actualGate[index]) ||
          std::fabs(actualGate[index] - gate) > gateTolerance ||
          !std::isfinite(actualUp[index]) ||
          std::fabs(actualUp[index] - up) > upTolerance) {
        std::fprintf(stderr,
                     "ggml_k_quant_swiglu_test: %s host projection oracle "
                     "mismatch at %zu\n",
                     name, index);
        return false;
      }
      const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
      const float oracle = (gate * sigmoid) * up;
      const float tolerance = 0.1f + 1.5e-2f * std::fabs(oracle);
      if (!std::isfinite(actual[index]) ||
          std::fabs(actual[index] - oracle) > tolerance) {
        std::fprintf(stderr,
                     "ggml_k_quant_swiglu_test: %s host-oracle mismatch at "
                     "%zu: actual=%g oracle=%g tolerance=%g\n",
                     name, index, static_cast<double>(actual[index]),
                     static_cast<double>(oracle),
                     static_cast<double>(tolerance));
        return false;
      }
    }
  }
  std::printf("%s batch=%lld M=%lld N=%lld K=%lld passed bitwise\n", name,
              static_cast<long long>(batchCount), static_cast<long long>(M),
              static_cast<long long>(N), static_cast<long long>(kK));
  return true;
}

} // namespace

int main() {
  const int device = hipdnn_ep::test::findGfx1151Device(kTestName);
  if (device == -1) {
    std::fprintf(stderr,
                 "ggml_k_quant_swiglu_test: no gfx1151 device; skipping\n");
    return 77;
  }
  if (device < 0 || !checkHip(hipSetDevice(device), "hipSetDevice"))
    return 1;

  hipStream_t stream = nullptr;
  if (!checkHip(hipStreamCreate(&stream), "hipStreamCreate"))
    return 1;
  const bool q4Decode = runCase<Q4KBlock, Q4KBlock>(
      stream, 1, 257, 1, HIP_KQUANT_FORMAT_Q4_K,
      HIP_KQUANT_FORMAT_Q4_K, "Q4/Q4 decode");
  const bool mixedBatchedDecode = runCase<Q4KBlock, Q6KBlock>(
      stream, 1, 67, 2, HIP_KQUANT_FORMAT_Q4_K,
      HIP_KQUANT_FORMAT_Q6_K, "Q4/Q6 batched decode");
  const bool reverseMixedDecode = runCase<Q6KBlock, Q4KBlock>(
      stream, 1, 67, 1, HIP_KQUANT_FORMAT_Q6_K,
      HIP_KQUANT_FORMAT_Q4_K, "Q6/Q4 decode");
  const bool q6Decode = runCase<Q6KBlock, Q6KBlock>(
      stream, 1, 31, 1, HIP_KQUANT_FORMAT_Q6_K,
      HIP_KQUANT_FORMAT_Q6_K, "Q6/Q6 decode");
  const bool q4Prefill = runCase<Q4KBlock, Q4KBlock>(
      stream, 40, 67, 1, HIP_KQUANT_FORMAT_Q4_K,
      HIP_KQUANT_FORMAT_Q4_K, "Q4/Q4 prefill fallback");
  const bool mixedBatchedPrefill = runCase<Q6KBlock, Q4KBlock>(
      stream, 3, 17, 2, HIP_KQUANT_FORMAT_Q6_K,
      HIP_KQUANT_FORMAT_Q4_K, "Q6/Q4 batched prefill fallback");
  const bool q6Prefill = runCase<Q6KBlock, Q6KBlock>(
      stream, 40, 31, 1, HIP_KQUANT_FORMAT_Q6_K,
      HIP_KQUANT_FORMAT_Q6_K, "Q6/Q6 prefill fallback");

  const bool destroyed = checkHip(hipStreamDestroy(stream), "stream destroy");
  return q4Decode && mixedBatchedDecode && reverseMixedDecode && q6Decode &&
                 q4Prefill && mixedBatchedPrefill && q6Prefill && destroyed
             ? 0
             : 1;
}
