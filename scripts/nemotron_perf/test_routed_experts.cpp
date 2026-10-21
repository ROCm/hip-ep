// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "routed_experts.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>

static decltype(&hip_routed_experts) frozenReference = nullptr;

static void check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Device {
  void *data = nullptr;
  explicit Device(size_t bytes) { check(hipMalloc(&data, bytes)); }
  ~Device() {
    if (data)
      (void)hipFree(data);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void upload(const void *source, size_t bytes) {
    check(hipMemcpy(data, source, bytes, hipMemcpyHostToDevice));
  }
};
static uint32_t seed = 13;
static uint32_t randomValue() { return seed = seed * 1664525u + 1013904223u; }
static float rounded(float value) { return __half2float(__float2half(value)); }

struct Matrix {
  int n, contraction, groups, rowBytes, zpBytes;
  std::vector<uint8_t> w, zp;
  std::vector<__half> scales;
  Device dw, dzp, ds;
  Matrix(int experts, int columns, int reduction, int block)
      : n(columns), contraction(reduction),
        groups((reduction + block - 1) / block), rowBytes(groups * block / 2),
        zpBytes((groups + 1) / 2),
        w(static_cast<size_t>(experts) * n * rowBytes),
        zp(static_cast<size_t>(experts) * n * zpBytes),
        scales(static_cast<size_t>(experts) * n * groups), dw(w.size()),
        dzp(zp.size()), ds(scales.size() * sizeof(__half)) {
    for (auto &v : w)
      v = randomValue() >> 24;
    for (auto &v : zp)
      v = randomValue() >> 24;
    for (auto &v : scales)
      v = __float2half(0.003f + (randomValue() % 7) * 0.001f);
    dw.upload(w.data(), w.size());
    dzp.upload(zp.data(), zp.size());
    ds.upload(scales.data(), scales.size() * sizeof(__half));
  }
  RoutedMatrixTable table(int experts, bool symmetric) {
    RoutedMatrixTable result = {};
    for (int e = 0; e < experts; ++e) {
      result.weights[e] = static_cast<uint8_t *>(dw.data) +
                          static_cast<size_t>(e) * n * rowBytes;
      result.scales[e] =
          static_cast<__half *>(ds.data) + static_cast<size_t>(e) * n * groups;
      result.zero_points[e] = symmetric
                                  ? nullptr
                                  : static_cast<uint8_t *>(dzp.data) +
                                        static_cast<size_t>(e) * n * zpBytes;
    }
    return result;
  }
  float dot(int expert, int column, const float *input, int block,
            bool symmetric, int mode) {
    const size_t row = static_cast<size_t>(expert) * n + column;
    double result = 0;
    if (mode == 1) {
      for (int begin = 0; begin < contraction; begin += 32) {
        int group = begin / block;
        float maximum = 0;
        for (int j = group * block;
             j < std::min((group + 1) * block, contraction); ++j)
          maximum = std::max(maximum, std::abs(input[j]));
        float inverse = maximum > 0 ? 127.0f / maximum : 0;
        float ascale = maximum > 0 ? maximum * (1.0f / 127.0f) : 1;
        int z = symmetric
                    ? 8
                    : (zp[row * zpBytes + group / 2] >> ((group & 1) * 4)) & 15;
        int dot = 0;
        for (int j = begin; j < begin + 32; ++j) {
          int q = (w[row * rowBytes + j / 2] >> ((j & 1) * 4)) & 15;
          int a =
              std::clamp(int(std::nearbyint(input[j] * inverse)), -127, 127);
          dot += a * (q - z);
        }
        result +=
            ascale * __half2float(scales[row * groups + group]) * float(dot);
      }
      return rounded(result);
    }
    for (int j = 0; j < contraction; ++j) {
      int q = (w[row * rowBytes + j / 2] >> ((j & 1) * 4)) & 15;
      int g = j / block;
      int z = symmetric ? 8 : (zp[row * zpBytes + g / 2] >> ((g & 1) * 4)) & 15;
      float weight = (q - z) * __half2float(scales[row * groups + g]);
      result += double(input[j]) * (mode >= 2 ? rounded(weight) : weight);
    }
    return rounded(result);
  }
};

#ifdef NEMOTRON_DENSE_REFERENCE
__global__ void reference_activation(__half *values, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = fmaxf(__half2float(values[i]), 0.0f);
    values[i] = __float2half(value * value);
  }
}

__global__ void reference_accumulate(const __half *values, __half *sum,
                                     const int64_t *indices, const float *gates,
                                     int rows, int columns, int k, int expert) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * columns)
    return;
  int token = i / columns;
  float gate = 0;
  for (int j = 0; j < k; ++j)
    if (indices[token * k + j] == expert)
      gate = gates[token * k + j];
  sum[i] = __hadd_rn(sum[i], __hmul_rn(values[i], __float2half(gate)));
}

// Run the original dense projection kernels on the same test inputs.
// These GPU calculations are diagnostics, not model CPU fallback.
static void compareDense(Matrix &up, Matrix &down, void *input, void *indices,
                         void *gates, const std::vector<__half> &actual,
                         int tokens, int experts, int k, int block,
                         bool symmetric, int mode, hipStream_t stream) {
  auto makeZeros = [](Matrix &matrix) {
    std::vector<uint8_t> unpacked(matrix.scales.size());
    for (size_t i = 0; i < unpacked.size(); ++i) {
      size_t row = i / matrix.groups, group = i % matrix.groups;
      unpacked[i] =
          (matrix.zp[row * matrix.zpBytes + group / 2] >> ((group & 1) * 4)) &
          15;
    }
    return unpacked;
  };
  auto uz = makeZeros(up), dz = makeZeros(down);
  std::vector<__half> uh(uz.size()), dh(dz.size());
  for (size_t i = 0; i < uz.size(); ++i)
    uh[i] = __float2half(uz[i]);
  for (size_t i = 0; i < dz.size(); ++i)
    dh[i] = __float2half(dz[i]);
  Device duz(uz.size()), ddz(dz.size()), duh(uh.size() * 2), ddh(dh.size() * 2);
  duz.upload(uz.data(), uz.size());
  ddz.upload(dz.data(), dz.size());
  duh.upload(uh.data(), uh.size() * 2);
  ddh.upload(dh.data(), dh.size() * 2);
  Device activation(size_t(tokens) * up.n * 2),
      value(size_t(tokens) * down.n * 2);
  Device sum(actual.size() * 2), quant(size_t(tokens) * std::max(up.n, down.n));
  Device qscales(size_t(tokens) * std::max(up.groups, down.groups) *
                 sizeof(float));
  check(hipMemsetAsync(sum.data, 0, actual.size() * 2, stream));
  auto ut = up.table(experts, symmetric), dt = down.table(experts, symmetric);
  auto project = [&](Matrix &matrix, RoutedMatrixTable &table, int e, void *a,
                     void *out, Device &z, Device &h) {
    size_t zeroOffset = size_t(e) * matrix.n * matrix.groups;
    void *zp =
        symmetric ? nullptr : static_cast<uint8_t *>(z.data) + zeroOffset;
    void *zh = symmetric ? nullptr : static_cast<__half *>(h.data) + zeroOffset;
    int rc;
    if (mode == 1)
      rc = hip_matmul_nbits_dp4a(stream, a, table.weights[e], table.scales[e],
                                 zp, nullptr, out, tokens, matrix.n,
                                 matrix.contraction, block, quant.data,
                                 qscales.data);
    else
      rc =
          hip_matmul_nbits(stream, a, table.weights[e], table.scales[e],
                           table.zero_points[e], nullptr, out, tokens, matrix.n,
                           matrix.contraction, 1, 4, block, 2, 1, zp, zh);
    check(static_cast<hipError_t>(rc));
  };
  for (int e = 0; e < experts; ++e) {
    project(up, ut, e, input, activation.data, duz, duh);
    hipLaunchKernelGGL(reference_activation, dim3((tokens * up.n + 255) / 256),
                       dim3(256), 0, stream,
                       static_cast<__half *>(activation.data), tokens * up.n);
    check(hipGetLastError());
    project(down, dt, e, activation.data, value.data, ddz, ddh);
    hipLaunchKernelGGL(reference_accumulate, dim3((actual.size() + 255) / 256),
                       dim3(256), 0, stream, static_cast<__half *>(value.data),
                       static_cast<__half *>(sum.data),
                       static_cast<const int64_t *>(indices),
                       static_cast<const float *>(gates), tokens, down.n, k, e);
    check(hipGetLastError());
  }
  check(hipStreamSynchronize(stream));
  std::vector<__half> reference(actual.size());
  check(hipMemcpy(reference.data(), sum.data, actual.size() * 2,
                  hipMemcpyDeviceToHost));
  double error2 = 0, ref2 = 0;
  float maxError = 0;
  size_t different = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    float a = __half2float(actual[i]), b = __half2float(reference[i]);
    if (!std::isfinite(a) || !std::isfinite(b))
      throw std::runtime_error("non-finite dense comparison");
    float error = a - b;
    different += a != b;
    error2 += double(error) * error;
    ref2 += double(b) * b;
    maxError = std::max(maxError, std::abs(error));
  }
  double relative = std::sqrt(error2 / std::max(ref2, 1e-30));
  std::printf("DENSE_REFERENCE T=%d H=%d I=%d mode=%d different=%zu/%zu "
              "max_abs=%.8g rel_l2=%.8g\n",
              tokens, down.n, up.n, mode, different, actual.size(), maxError,
              relative);
  if (relative > 0.006)
    throw std::runtime_error("dense reference tolerance exceeded");
}
#endif

static void test(int tokens, int hidden, int intermediate, int experts, int k,
                 int block, bool symmetric, int mode = 0) {
  Matrix up(experts, intermediate, hidden, block);
  Matrix down(experts, hidden, intermediate, block);
  auto upTable = up.table(experts, symmetric);
  auto downTable = down.table(experts, symmetric);
  std::vector<__half> input(tokens * hidden), output(tokens * hidden);
  std::vector<int64_t> indices(tokens * k);
  std::vector<float> gates(tokens * k);
  for (auto &v : input)
    v = __float2half((int(randomValue() % 201) - 100) * 0.002f);
  std::vector<int> permutation(experts);
  for (int token = 0; token < tokens; ++token) {
    for (int e = 0; e < experts; ++e)
      permutation[e] = e;
    for (int e = experts - 1; e > 0; --e)
      std::swap(permutation[e], permutation[randomValue() % (e + 1)]);
    for (int slot = 0; slot < k; ++slot) {
      indices[token * k + slot] = permutation[slot];
      gates[token * k + slot] = (randomValue() % 100) * 0.01f;
    }
  }
  Device din(input.size() * sizeof(__half)),
      didx(indices.size() * sizeof(int64_t));
  Device dgates(gates.size() * sizeof(float)),
      dact(tokens * k * intermediate * sizeof(__half));
  Device dslots(tokens * k * hidden * sizeof(__half)),
      dout(output.size() * sizeof(__half));
  Device dquant(
      std::max(size_t(std::max(tokens * hidden, tokens * k * intermediate)),
               routed_grouped_scratch_bytes(tokens, k, experts)));
  Device dqscale(std::max(tokens * ((hidden + block - 1) / block),
                          tokens * k * ((intermediate + block - 1) / block)) *
                 sizeof(float));
  din.upload(input.data(), input.size() * sizeof(__half));
  didx.upload(indices.data(), indices.size() * sizeof(int64_t));
  dgates.upload(gates.data(), gates.size() * sizeof(float));
  hipStream_t stream;
  check(hipStreamCreate(&stream));
  auto launchWith = [&](decltype(&hip_routed_experts) kernel,
                        const int64_t *indexOverride = nullptr) {
    check(static_cast<hipError_t>(kernel(
        stream, din.data,
        indexOverride ? indexOverride : static_cast<int64_t *>(didx.data),
        static_cast<float *>(dgates.data), &upTable, &downTable, dact.data,
        dslots.data, dout.data, tokens, hidden, intermediate, experts, k, block,
        mode, dquant.data, static_cast<float *>(dqscale.data))));
  };
  auto launch = [&] { launchWith(hip_routed_experts); };
  launch();
  check(hipStreamSynchronize(stream));
  check(hipMemcpy(output.data(), dout.data, output.size() * sizeof(__half),
                  hipMemcpyDeviceToHost));
  if (frozenReference) {
    launchWith(frozenReference);
    check(hipStreamSynchronize(stream));
    std::vector<__half> expected(output.size());
    check(hipMemcpy(expected.data(), dout.data,
                    expected.size() * sizeof(__half), hipMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), output.data(),
                    output.size() * sizeof(__half)))
      throw std::runtime_error("frozen sparse GPU reference differs");
    std::printf("FROZEN_REFERENCE T=%d H=%d I=%d mode=%d bit_exact=%zu\n",
                tokens, hidden, intermediate, mode, output.size());
  }
  if (tokens >= 128 && !frozenReference)
    throw std::runtime_error("large test requires the frozen GPU reference");
  double error2 = 0, reference2 = 0;
  float maxError = 0, maxReference = 0;
  for (int token = 0; token < (tokens >= 128 ? 0 : tokens); ++token) {
    std::vector<float> x(hidden), act(intermediate), ref(hidden, 0.0f);
    for (int j = 0; j < hidden; ++j)
      x[j] = __half2float(input[token * hidden + j]);
    // This CPU oracle checks the test kernel. It is not a model fallback.
    for (int e = 0; e < experts; ++e) {
      int slot = -1;
      for (int s = 0; s < k; ++s)
        if (indices[token * k + s] == e)
          slot = s;
      if (slot < 0)
        continue;
      for (int n = 0; n < intermediate; ++n) {
        float value =
            std::max(up.dot(e, n, x.data(), block, symmetric, mode), 0.0f);
        act[n] = rounded(value * value);
      }
      for (int n = 0; n < hidden; ++n) {
        float value = down.dot(e, n, act.data(), block, symmetric, mode);
        ref[n] =
            rounded(ref[n] + rounded(value * rounded(gates[token * k + slot])));
      }
    }
    for (int n = 0; n < hidden; ++n) {
      const float actual = __half2float(output[token * hidden + n]);
      if (!std::isfinite(actual))
        throw std::runtime_error("non-finite output");
      float error = std::abs(actual - ref[n]);
      maxError = std::max(maxError, error);
      maxReference = std::max(maxReference, std::abs(ref[n]));
      error2 += double(error) * error;
      reference2 += double(ref[n]) * ref[n];
    }
  }
  double relative = std::sqrt(error2 / std::max(reference2, 1e-30));
  if (relative > 0.006 || maxError > std::max(0.0001f, maxReference * 0.01f))
    throw std::runtime_error("numerical tolerance exceeded");
#ifdef NEMOTRON_DENSE_REFERENCE
  // The existing DP4A kernel uses K/2 as its weight-row stride. Do not use it
  // as an oracle for padded rows. The CPU oracle still checks those cases.
  if ((mode == 1 || (mode == 3 && tokens >= 8)) && hidden % block == 0 &&
      intermediate % block == 0)
    compareDense(up, down, din.data, didx.data, dgates.data, output, tokens,
                 experts, k, block, symmetric, mode, stream);
#endif
  // Warm up after the CPU check. Time GPU kernels only.
  launch();
  hipEvent_t start, end;
  check(hipEventCreate(&start));
  check(hipEventCreate(&end));
  check(hipEventRecord(start, stream));
  for (int repeat = 0; repeat < 20; ++repeat)
    launch();
  check(hipEventRecord(end, stream));
  check(hipEventSynchronize(end));
  float elapsed;
  check(hipEventElapsedTime(&elapsed, start, end));
  if (frozenReference) {
    launchWith(frozenReference);
    check(hipEventRecord(start, stream));
    for (int repeat = 0; repeat < 20; ++repeat)
      launchWith(frozenReference);
    check(hipEventRecord(end, stream));
    check(hipEventSynchronize(end));
    float referenceMs;
    check(hipEventElapsedTime(&referenceMs, start, end));
    std::printf(
        "FROZEN_TIMING T=%d mode=%d candidate_ms=%.6f reference_ms=%.6f\n",
        tokens, mode, elapsed / 20, referenceMs / 20);
  }
  if (tokens == 1 && hidden == 2688 && mode == 1 && frozenReference) {
    // Sixteen disjoint expert sets read much more than the 32 MiB cache.
    // Each set has its own device index array. Timings exclude host copies.
    constexpr int rotations = 16;
    if (experts < rotations * k)
      throw std::runtime_error("not enough experts for cold rotation");
    std::vector<int64_t> rotationIndices(rotations * k);
    for (int i = 0; i < rotations * k; ++i)
      rotationIndices[i] = i;
    Device drotation(rotationIndices.size() * sizeof(int64_t));
    drotation.upload(rotationIndices.data(),
                     rotationIndices.size() * sizeof(int64_t));
    auto indicesFor = [&](int i) {
      return static_cast<const int64_t *>(drotation.data) + i * k;
    };
    std::vector<__half> actual(output.size()), expected(output.size());
    for (int i = 0; i < rotations; ++i) {
      launchWith(hip_routed_experts, indicesFor(i));
      check(hipStreamSynchronize(stream));
      check(hipMemcpy(actual.data(), dout.data, actual.size() * sizeof(__half),
                      hipMemcpyDeviceToHost));
      launchWith(frozenReference, indicesFor(i));
      check(hipStreamSynchronize(stream));
      check(hipMemcpy(expected.data(), dout.data,
                      expected.size() * sizeof(__half), hipMemcpyDeviceToHost));
      if (std::memcmp(actual.data(), expected.data(),
                      actual.size() * sizeof(__half)))
        throw std::runtime_error("cold rotation differs from frozen reference");
    }
    auto coldTime = [&](decltype(&hip_routed_experts) kernel) {
      for (int i = 0; i < rotations; ++i)
        launchWith(kernel, indicesFor(i));
      check(hipEventRecord(start, stream));
      for (int i = 0; i < rotations; ++i)
        launchWith(kernel, indicesFor(i));
      check(hipEventRecord(end, stream));
      check(hipEventSynchronize(end));
      float ms;
      check(hipEventElapsedTime(&ms, start, end));
      return ms / rotations;
    };
    // Reverse the order each round to expose clock or thermal drift.
    for (int round = 0; round < 4; ++round) {
      float candidate, reference;
      if (round % 2) {
        reference = coldTime(frozenReference);
        candidate = coldTime(hip_routed_experts);
      } else {
        candidate = coldTime(hip_routed_experts);
        reference = coldTime(frozenReference);
      }
      std::printf("FROZEN_COLD_TIMING round=%d sets=%d bit_exact=1 "
                  "candidate_ms=%.6f reference_ms=%.6f\n",
                  round, rotations, candidate, reference);
    }
  }
  std::printf("PASS T=%d H=%d I=%d E=%d k=%d bs=%d symmetric=%d mode=%d "
              "max_abs=%.8g rel_l2=%.8g kernel_ms=%.6f\n",
              tokens, hidden, intermediate, experts, k, block, symmetric, mode,
              maxError, relative, elapsed / 20);
  check(hipEventDestroy(start));
  check(hipEventDestroy(end));
  check(hipStreamDestroy(stream));
}

int main() {
  try {
    if (const char *path = std::getenv("NEMOTRON_SPARSE_REFERENCE_LIB")) {
      void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
      if (!handle)
        throw std::runtime_error(dlerror());
      frozenReference = reinterpret_cast<decltype(frozenReference)>(
          dlsym(handle, "hip_routed_experts"));
      if (!frozenReference)
        throw std::runtime_error(dlerror());
    }
    if (const char *cold = std::getenv("NEMOTRON_ROUTED_COLD_ONLY");
        cold && std::strcmp(cold, "1") == 0) {
      test(1, 2688, 1856, 128, 6, 64, false, 1);
      return 0;
    }
    test(1, 64, 96, 4, 2, 32, false);
    test(3, 65, 33, 7, 3, 32, false);
    test(2, 257, 129, 8, 4, 64, false);
    test(1, 64, 96, 4, 4, 128, true);
    test(1, 2688, 1856, 128, 6, 64, false);
    test(3, 64, 96, 7, 3, 32, false, 1);
    test(2, 96, 64, 4, 4, 128, true, 1);
    test(1, 2688, 1856, 128, 6, 64, false, 1);
    test(3, 65, 33, 7, 3, 32, false, 2);
    test(1, 2688, 1856, 128, 6, 64, false, 2);
    test(33, 64, 96, 7, 3, 32, false, 3);
    test(17, 96, 64, 4, 4, 128, true, 3);
    test(1, 2688, 1856, 128, 6, 64, false, 3);
    test(32, 2688, 1856, 128, 6, 64, false, 3);
    test(8, 2688, 1856, 128, 6, 64, false, 3);
    if (frozenReference) {
      test(128, 2688, 1856, 128, 6, 64, false, 3);
      test(512, 2688, 1856, 128, 6, 64, false, 3);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
