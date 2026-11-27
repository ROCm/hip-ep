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

// The fused W2A8 dp4a decode (hip_qmoe_decode_fused_dp4a_u2): fc1 + SwiGLU,
// fc2 and the routing-weighted reduce for a burst of tokens.
namespace decode {

constexpr int kHidden = 256;
constexpr int kInter = 128;
constexpr int kTopK = 2;
constexpr int kDecodeExperts = 4;
constexpr float kAlpha = 1.702f;
constexpr float kBeta = 1.0f;
constexpr float kLimit = 7.0f;

template <typename T> bool scratch(T **device, size_t count) {
  return hipMalloc(device, count * sizeof(T)) == hipSuccess;
}

// One expert matrix: rows x cols 2-bit weights, a scale per (row, group) and
// 2-bit zero points in the ONNX packing.
struct Matrix {
  int rows, cols, blocks, zpRowBytes;
  std::vector<uint8_t> weights, zeroPoints, packedZeroPoints;
  std::vector<__half> scales, bias;

  Matrix(int r, int c, int blockSize, int seed)
      : rows(r), cols(c), blocks(c / blockSize), zpRowBytes((blocks + 3) / 4),
        weights(static_cast<size_t>(kDecodeExperts) * r * c / 4),
        zeroPoints(static_cast<size_t>(kDecodeExperts) * r * blocks),
        packedZeroPoints(static_cast<size_t>(kDecodeExperts) * r * zpRowBytes,
                         0),
        scales(zeroPoints.size()),
        bias(static_cast<size_t>(kDecodeExperts) * r) {
    for (size_t i = 0; i < weights.size(); ++i)
      weights[i] = static_cast<uint8_t>((i * 37 + seed * 11 + 13) & 0xff);
    // Scales spread 1x..5x and zero points cover 0..3, so a value read from
    // the wrong group shows up well above the activation quantization error.
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] =
          __float2half(0.01f * static_cast<float>(1 + (i * 3 + seed) % 5));
      zeroPoints[i] = static_cast<uint8_t>((i * 5 + seed) & 3);
    }
    for (int row = 0; row < kDecodeExperts * r; ++row)
      for (int g = 0; g < blocks; ++g)
        packedZeroPoints[row * zpRowBytes + g / 4] |= static_cast<uint8_t>(
            zeroPoints[row * blocks + g] << ((g % 4) * 2));
    for (size_t i = 0; i < bias.size(); ++i)
      bias[i] = __float2half(static_cast<float>(static_cast<int>(i % 7) - 3) /
                             16.0f);
  }

  float dequant(int expert, int row, int col, bool withZeroPoints) const {
    const size_t r = static_cast<size_t>(expert) * rows + row;
    const uint8_t byte = weights[r * (cols / 4) + col / 4];
    const float q = static_cast<float>((byte >> ((col & 3) * 2)) & 3u);
    const size_t p = r * blocks + col / (cols / blocks);
    const float zp = withZeroPoints ? static_cast<float>(zeroPoints[p]) : 2.0f;
    return (q - zp) * __half2float(scales[p]);
  }
};

// 32-element groups split every 64-element chunk of the dot in two halves.
bool runCase(int blockSize, int tokens, bool withZeroPoints) {
  const int slots = tokens * kTopK;
  Matrix fc1(2 * kInter, kHidden, blockSize, 1);
  Matrix fc2(kHidden, kInter, blockSize, 2);

  std::vector<__half> input(static_cast<size_t>(tokens) * kHidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = __float2half(
        static_cast<float>(static_cast<int>(i * 7 % 23) - 11) / 12.0f);
  std::vector<int32_t> expertIndices(slots);
  std::vector<__half> expertWeights(slots);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < kTopK; ++j) {
      expertIndices[t * kTopK + j] = (t + 1 + j * 2) % kDecodeExperts;
      expertWeights[t * kTopK + j] = __float2half(j == 0 ? 0.625f : 0.375f);
    }

  // fc1 rows interleave gate (2i) and linear (2i + 1).
  std::vector<float> expected(static_cast<size_t>(tokens) * kHidden, 0.0f);
  for (int s = 0; s < slots; ++s) {
    const int t = s / kTopK, e = expertIndices[s];
    std::vector<float> act(kInter);
    for (int i = 0; i < kInter; ++i) {
      float v[2];
      for (int h = 0; h < 2; ++h) {
        const int row = 2 * i + h;
        float sum = __half2float(fc1.bias[e * fc1.rows + row]);
        for (int c = 0; c < kHidden; ++c)
          sum += __half2float(input[t * kHidden + c]) *
                 fc1.dequant(e, row, c, withZeroPoints);
        v[h] = sum;
      }
      const float g = std::min(v[0], kLimit);
      const float l = std::max(std::min(v[1], kLimit), -kLimit);
      act[i] = g / (1.0f + std::exp(-kAlpha * g)) * (l + kBeta);
    }
    for (int n = 0; n < kHidden; ++n) {
      float sum = __half2float(fc2.bias[e * fc2.rows + n]);
      for (int c = 0; c < kInter; ++c)
        sum += act[c] * fc2.dequant(e, n, c, withZeroPoints);
      expected[t * kHidden + n] += __half2float(expertWeights[s]) * sum;
    }
  }

  __half *dInput = nullptr, *dExpertWeights = nullptr, *dFc1Scales = nullptr,
         *dFc1Bias = nullptr, *dFc2Scales = nullptr, *dFc2Bias = nullptr,
         *dOutput = nullptr;
  uint8_t *dFc1 = nullptr, *dFc1Zp = nullptr, *dFc2 = nullptr,
          *dFc2Zp = nullptr;
  int32_t *dExpertIndices = nullptr, *dSumIn = nullptr, *dSumMid = nullptr;
  int8_t *dQbIn = nullptr, *dQbMid = nullptr;
  float *dSlotBuf = nullptr, *dAct = nullptr, *dScaleIn = nullptr,
        *dScaleMid = nullptr;
  bool ok =
      upload(&dInput, input) && upload(&dExpertIndices, expertIndices) &&
      upload(&dExpertWeights, expertWeights) && upload(&dFc1, fc1.weights) &&
      upload(&dFc1Scales, fc1.scales) &&
      upload(&dFc1Zp, fc1.packedZeroPoints) && upload(&dFc1Bias, fc1.bias) &&
      upload(&dFc2, fc2.weights) && upload(&dFc2Scales, fc2.scales) &&
      upload(&dFc2Zp, fc2.packedZeroPoints) && upload(&dFc2Bias, fc2.bias) &&
      scratch(&dOutput, expected.size()) &&
      scratch(&dSlotBuf, static_cast<size_t>(slots) * kHidden) &&
      scratch(&dAct, static_cast<size_t>(slots) * kInter) &&
      scratch(&dQbIn, static_cast<size_t>(tokens) * kHidden) &&
      scratch(&dScaleIn, static_cast<size_t>(tokens) * kHidden / blockSize) &&
      scratch(&dSumIn, static_cast<size_t>(tokens) * kHidden / 32) &&
      scratch(&dQbMid, static_cast<size_t>(slots) * kInter) &&
      scratch(&dScaleMid, static_cast<size_t>(slots) * kInter / blockSize) &&
      scratch(&dSumMid, static_cast<size_t>(slots) * kInter / 32);
  hipStream_t stream = nullptr;
  if (ok)
    ok = hipStreamCreate(&stream) == hipSuccess;
  if (ok) {
    const int status = hip_qmoe_decode_fused_dp4a_u2(
        stream, dInput, dExpertIndices, dExpertWeights, dFc1, dFc1Scales,
        withZeroPoints ? dFc1Zp : nullptr, dFc1Bias, dFc2, dFc2Scales,
        withZeroPoints ? dFc2Zp : nullptr, dFc2Bias, dSlotBuf, dAct, dOutput,
        dQbIn, dScaleIn, dSumIn, dQbMid, dScaleMid, dSumMid, kHidden, kInter,
        kTopK, blockSize, kAlpha, kBeta, kLimit, tokens);
    ok = status == static_cast<int>(hipSuccess) &&
         hipStreamSynchronize(stream) == hipSuccess;
  }

  std::vector<__half> actual(expected.size());
  if (ok)
    ok = hipMemcpy(actual.data(), dOutput, actual.size() * sizeof(__half),
                   hipMemcpyDeviceToHost) == hipSuccess;
  // Both activations are quantized to int8 per group, so the result carries
  // well under 1% error against the fp32 reference.
  double errSq = 0.0, refSq = 0.0;
  if (ok) {
    for (size_t i = 0; i < actual.size(); ++i) {
      const double d = __half2float(actual[i]) - expected[i];
      errSq += d * d;
      refSq += static_cast<double>(expected[i]) * expected[i];
    }
  }
  const double relL2 = refSq > 0.0 ? std::sqrt(errSq / refSq) : 1.0;
  ok = ok && std::isfinite(relL2) && relL2 <= 0.03;

  if (stream)
    hipStreamDestroy(stream);
  for (void *p :
       {static_cast<void *>(dInput), static_cast<void *>(dExpertWeights),
        static_cast<void *>(dFc1Scales), static_cast<void *>(dFc1Bias),
        static_cast<void *>(dFc2Scales), static_cast<void *>(dFc2Bias),
        static_cast<void *>(dOutput), static_cast<void *>(dFc1),
        static_cast<void *>(dFc1Zp), static_cast<void *>(dFc2),
        static_cast<void *>(dFc2Zp), static_cast<void *>(dExpertIndices),
        static_cast<void *>(dSumIn), static_cast<void *>(dSumMid),
        static_cast<void *>(dQbIn), static_cast<void *>(dQbMid),
        static_cast<void *>(dSlotBuf), static_cast<void *>(dAct),
        static_cast<void *>(dScaleIn), static_cast<void *>(dScaleMid)})
    hipFree(p);

  std::cout << (ok ? "PASS" : "FAIL") << " decode block_size=" << blockSize
            << " tokens=" << tokens << " zero_points=" << withZeroPoints
            << " rel_l2=" << relL2 << '\n';
  return ok;
}

// Geometry the kernel cannot serve must be refused before any launch.
bool rejects(int64_t hidden, int64_t inter, int64_t blockSize) {
  const int status = hip_qmoe_decode_fused_dp4a_u2(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, hidden, inter, kTopK,
      blockSize, kAlpha, kBeta, kLimit, 1);
  const bool ok = status == -1;
  std::cout << (ok ? "PASS" : "FAIL") << " decode rejects hidden=" << hidden
            << " inter=" << inter << " block_size=" << blockSize << '\n';
  return ok;
}

} // namespace decode

} // namespace

int main() {
  bool ok = true;
  for (int rows : {2, 3, 20, 64})
    for (int k : {96, 80})
      for (bool gather : {false, true})
        for (bool zeroPoints : {false, true})
          ok = runCase(rows, k, gather, zeroPoints) && ok;
  for (int blockSize : {32, 64, 128})
    for (int tokens : {1, 3})
      for (bool zeroPoints : {false, true})
        ok = decode::runCase(blockSize, tokens, zeroPoints) && ok;
  ok = decode::rejects(decode::kHidden, decode::kInter, 16) && ok;
  ok = decode::rejects(decode::kHidden, 96, 32) && ok;
  ok = decode::rejects(decode::kHidden, decode::kInter, 48) && ok;
  return ok ? 0 : 1;
}
