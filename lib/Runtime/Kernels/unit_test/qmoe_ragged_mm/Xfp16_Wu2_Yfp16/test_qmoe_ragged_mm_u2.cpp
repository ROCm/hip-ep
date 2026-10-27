/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int kRows = 3;
constexpr int kExperts = 2;
constexpr int kN = 64;
constexpr int kK = 64;
constexpr int kBlockSize = 32;
constexpr int kBits = 2;

template <typename T> bool upload(T **device, const std::vector<T> &host) {
  if (hipMalloc(device, host.size() * sizeof(T)) != hipSuccess)
    return false;
  return hipMemcpy(*device, host.data(), host.size() * sizeof(T),
                   hipMemcpyHostToDevice) == hipSuccess;
}

uint8_t packedValue(const std::vector<uint8_t> &weights, int expert, int n,
                    int k) {
  constexpr int rowBytes = kK / 4;
  const size_t row = static_cast<size_t>(expert * kN + n) * rowBytes;
  const uint8_t byte = weights[row + k / 4];
  return (byte >> ((k & 3) * 2)) & 3u;
}

bool runCase(bool gather, bool withZeroPoints) {
  constexpr int kBlocks = kK / kBlockSize;
  constexpr int rowBytes = kK / 4;
  std::vector<__half> input(kRows * kK);
  std::vector<uint8_t> weights(kExperts * kN * rowBytes);
  std::vector<__half> scales(kExperts * kN * kBlocks);
  std::vector<uint8_t> zeroPoints(kExperts * kN * kBlocks);
  std::vector<__half> bias(kExperts * kN);
  const std::vector<int32_t> rowIds = {2, 0, 1};
  const std::vector<int32_t> groups = {1, 0, kRows, 0};
  const std::vector<int32_t> groupCount = {1};

  for (size_t i = 0; i < input.size(); ++i)
    input[i] =
        __float2half(static_cast<float>(static_cast<int>(i % 11) - 5) / 8.0f);
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<uint8_t>((i * 37 + 13) & 0xff);
  for (size_t i = 0; i < scales.size(); ++i)
    scales[i] = __float2half(0.02f + 0.001f * static_cast<float>(i % 7));
  for (size_t i = 0; i < zeroPoints.size(); ++i)
    zeroPoints[i] = static_cast<uint8_t>(i & 3);
  for (size_t i = 0; i < bias.size(); ++i)
    bias[i] =
        __float2half(static_cast<float>(static_cast<int>(i % 5) - 2) / 32.0f);

  std::vector<__half> expected(kRows * kN);
  for (int r = 0; r < kRows; ++r) {
    const int inputRow = gather ? rowIds[r] : r;
    for (int n = 0; n < kN; ++n) {
      float sum = __half2float(bias[kN + n]);
      for (int k = 0; k < kK; ++k) {
        const int group = k / kBlockSize;
        const float q = packedValue(weights, 1, n, k);
        const size_t param = static_cast<size_t>(kN + n) * kBlocks + group;
        const float zp =
            withZeroPoints ? static_cast<float>(zeroPoints[param]) : 2.0f;
        sum += __half2float(input[inputRow * kK + k]) * (q - zp) *
               __half2float(scales[param]);
      }
      expected[r * kN + n] = __float2half(sum);
    }
  }

  __half *dInput = nullptr, *dScales = nullptr, *dBias = nullptr,
         *dOutput = nullptr;
  uint8_t *dWeights = nullptr, *dZeroPoints = nullptr;
  int32_t *dRowIds = nullptr, *dGroups = nullptr, *dGroupCount = nullptr;
  uint32_t *dQueue = nullptr;
  bool ok = upload(&dInput, input) && upload(&dWeights, weights) &&
            upload(&dScales, scales) && upload(&dZeroPoints, zeroPoints) &&
            upload(&dBias, bias) && upload(&dRowIds, rowIds) &&
            upload(&dGroups, groups) && upload(&dGroupCount, groupCount) &&
            hipMalloc(&dQueue, sizeof(uint32_t)) == hipSuccess &&
            hipMalloc(&dOutput, expected.size() * sizeof(__half)) == hipSuccess;
  hipStream_t stream = nullptr;
  if (ok)
    ok = hipStreamCreate(&stream) == hipSuccess &&
         hipMemset(dQueue, 0, sizeof(uint32_t)) == hipSuccess;
  if (ok) {
    const int status = hip_qmoe_ragged_matmul_nbits(
        stream, dInput, gather ? dRowIds : nullptr, dGroups, dGroupCount,
        dQueue, dWeights, dScales, withZeroPoints ? dZeroPoints : nullptr,
        dBias, dOutput, kRows, kExperts, kN, kK, kBits, kBlockSize,
        sizeof(__half));
    ok = status == static_cast<int>(hipSuccess) &&
         hipStreamSynchronize(stream) == hipSuccess;
  }

  std::vector<__half> actual(expected.size());
  if (ok)
    ok = hipMemcpy(actual.data(), dOutput, actual.size() * sizeof(__half),
                   hipMemcpyDeviceToHost) == hipSuccess;
  float maxError = 0.0f;
  if (ok) {
    for (size_t i = 0; i < actual.size(); ++i)
      maxError = std::max(maxError, std::abs(__half2float(actual[i]) -
                                             __half2float(expected[i])));
    ok = std::isfinite(maxError) && maxError <= 0.03f;
  }

  if (stream)
    hipStreamDestroy(stream);
  hipFree(dInput);
  hipFree(dWeights);
  hipFree(dScales);
  hipFree(dZeroPoints);
  hipFree(dBias);
  hipFree(dRowIds);
  hipFree(dGroups);
  hipFree(dGroupCount);
  hipFree(dQueue);
  hipFree(dOutput);

  std::cout << (ok ? "PASS" : "FAIL") << " gather=" << gather
            << " zero_points=" << withZeroPoints << " max_error=" << maxError
            << '\n';
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = runCase(false, false) && ok;
  ok = runCase(true, false) && ok;
  ok = runCase(false, true) && ok;
  ok = runCase(true, true) && ok;
  return ok ? 0 : 1;
}
