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

constexpr int kExperts = 2;
constexpr int kN = 64;
constexpr int kBlockSize = 64;
constexpr int kBits = 2;

template <typename T> bool upload(T **device, const std::vector<T> &host) {
  if (hipMalloc(device, host.size() * sizeof(T)) != hipSuccess)
    return false;
  return hipMemcpy(*device, host.data(), host.size() * sizeof(T),
                   hipMemcpyHostToDevice) == hipSuccess;
}

uint8_t packedValue(const std::vector<uint8_t> &weights, int kK, int expert,
                    int n, int k) {
  const int rowBytes = (kK + 3) / 4;
  const size_t row = static_cast<size_t>(expert * kN + n) * rowBytes;
  const uint8_t byte = weights[row + k / 4];
  return (byte >> ((k & 3) * 2)) & 3u;
}

// kRows covers each task of the ragged kernel for 2-bit groups: up to 2 rows
// run the GEMV, up to 31 the 16-row WMMA tile (20 rows take two of them), more
// the 64-row tile. A kK that is not a multiple of 32 exercises the K tail.
bool runCase(int kRows, int kK, bool gather, bool withZeroPoints) {
  const int kBlocks = (kK + kBlockSize - 1) / kBlockSize;
  const int rowBytes = (kK + 3) / 4;
  std::vector<__half> input(kRows * kK);
  std::vector<uint8_t> weights(kExperts * kN * rowBytes);
  std::vector<__half> scales(kExperts * kN * kBlocks);
  std::vector<uint8_t> zeroPoints(kExperts * kN * kBlocks);
  std::vector<__half> bias(kExperts * kN);
  std::vector<int32_t> rowIds(kRows);
  for (int r = 0; r < kRows; ++r)
    rowIds[r] = (r * 7 + 2) % kRows;
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
  // The kernel takes 2-bit zero points in the ONNX packing: four per byte,
  // low bits first, each (expert, n) row padded to a whole byte.
  const int zpRowBytes = (kBlocks * kBits + 7) / 8;
  std::vector<uint8_t> packedZeroPoints(kExperts * kN * zpRowBytes, 0);
  for (int row = 0; row < kExperts * kN; ++row)
    for (int g = 0; g < kBlocks; ++g)
      packedZeroPoints[row * zpRowBytes + g / 4] |= static_cast<uint8_t>(
          zeroPoints[row * kBlocks + g] << ((g % 4) * kBits));
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
        const float q = packedValue(weights, kK, 1, n, k);
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
            upload(&dScales, scales) &&
            upload(&dZeroPoints, packedZeroPoints) &&
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

  std::cout << (ok ? "PASS" : "FAIL") << " rows=" << kRows << " K=" << kK
            << " gather=" << gather
            << " zero_points=" << withZeroPoints << " max_error=" << maxError
            << '\n';
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  for (int rows : {2, 3, 20, 64})
    for (int k : {96, 80})
      for (bool gather : {false, true})
        for (bool zeroPoints : {false, true})
          ok = runCase(rows, k, gather, zeroPoints) && ok;
  return ok ? 0 : 1;
}
