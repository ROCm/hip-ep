/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

class Stream {
public:
  Stream() : stream_(nullptr), status_(hipStreamCreate(&stream_)) {}
  ~Stream() {
    if (stream_)
      (void)hipStreamDestroy(stream_);
  }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  hipError_t status() const { return status_; }
  hipStream_t get() const { return stream_; }

private:
  hipStream_t stream_;
  hipError_t status_;
};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) : pointer_(nullptr) {
    status_ =
        hipMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T));
  }
  ~DeviceBuffer() {
    if (pointer_)
      (void)hipFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  hipError_t status() const { return status_; }
  T *get() const { return pointer_; }

private:
  T *pointer_;
  hipError_t status_;
};

bool checkHip(hipError_t status, const char *operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

uint32_t floatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float inputValue(size_t index) {
  const int value = static_cast<int>((index * 17 + 11) % 257) - 128;
  return static_cast<float>(value) / 64.0f;
}

float weightValue(size_t output, size_t index) {
  const int value =
      static_cast<int>((output * 29 + index * 13 + 7) % 251) - 125;
  return static_cast<float>(value) / 96.0f;
}

bool runCase(int M, int N, int K, bool requireExact) {
  std::vector<__hip_bfloat16> input(static_cast<size_t>(M) * K);
  std::vector<__hip_bfloat16> weights(static_cast<size_t>(N) * K);
  std::vector<float> expected(static_cast<size_t>(M) * N);
  std::vector<float> actual(expected.size());

  for (size_t index = 0; index < input.size(); ++index) {
    const float value =
        requireExact ? static_cast<float>((index % 5) - 2) : inputValue(index);
    input[index] = __float2bfloat16(value);
  }
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const float value = requireExact ? static_cast<float>((n % 7) - 3)
                                       : weightValue(static_cast<size_t>(n), k);
      weights[static_cast<size_t>(n) * K + k] = __float2bfloat16(value);
    }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k)
        sum += static_cast<double>(
                   static_cast<float>(input[static_cast<size_t>(m) * K + k])) *
               static_cast<float>(weights[static_cast<size_t>(n) * K + k]);
      expected[static_cast<size_t>(m) * N + n] = static_cast<float>(sum);
    }

  Stream stream;
  DeviceBuffer<__hip_bfloat16> deviceInput(input.size() + 8);
  DeviceBuffer<__hip_bfloat16> deviceWeights(weights.size() + 8);
  DeviceBuffer<float> deviceOutput(actual.size());
  if (!checkHip(stream.status(), "hipStreamCreate") ||
      !checkHip(deviceInput.status(), "hipMalloc(input)") ||
      !checkHip(deviceWeights.status(), "hipMalloc(weights)") ||
      !checkHip(deviceOutput.status(), "hipMalloc(output)"))
    return false;
  if (!checkHip(hipMemcpyAsync(deviceInput.get(), input.data(),
                               input.size() * sizeof(input.front()),
                               hipMemcpyHostToDevice, stream.get()),
                "hipMemcpyAsync(input)") ||
      !checkHip(hipMemcpyAsync(deviceWeights.get(), weights.data(),
                               weights.size() * sizeof(weights.front()),
                               hipMemcpyHostToDevice, stream.get()),
                "hipMemcpyAsync(weights)"))
    return false;

  // Operands must be 16-byte aligned and K a multiple of 8.
  const auto *misalignedInput = deviceInput.get() + 4;
  const auto *misalignedWeights = deviceWeights.get() + 4;
  if (hip_gemm_bf16_f32_transposed_b(stream.get(), misalignedInput,
                                     deviceWeights.get(), deviceOutput.get(), M,
                                     N, K) != -1 ||
      hip_gemm_bf16_f32_transposed_b(stream.get(), deviceInput.get(),
                                     misalignedWeights, deviceOutput.get(), M,
                                     N, K) != -1 ||
      hip_gemm_bf16_f32_transposed_b(stream.get(), deviceInput.get(),
                                     deviceWeights.get(), deviceOutput.get(), M,
                                     N, K - 4) != -1) {
    std::fprintf(stderr, "hip_gemm_bf16_f32_transposed_b accepted an "
                         "unsupported operand\n");
    return false;
  }

  if (hip_gemm_bf16_f32_transposed_b(stream.get(), deviceInput.get(),
                                     deviceWeights.get(), deviceOutput.get(), M,
                                     N, K) != 0) {
    std::fprintf(stderr, "hip_gemm_bf16_f32_transposed_b launch failed\n");
    return false;
  }
  if (!checkHip(hipMemcpyAsync(actual.data(), deviceOutput.get(),
                               actual.size() * sizeof(actual.front()),
                               hipMemcpyDeviceToHost, stream.get()),
                "hipMemcpyAsync(output)") ||
      !checkHip(hipStreamSynchronize(stream.get()), "hipStreamSynchronize"))
    return false;

  for (size_t index = 0; index < expected.size(); ++index) {
    const float reference = expected[index];
    const float value = actual[index];
    const float difference = std::abs(value - reference);
    const float tolerance =
        requireExact ? 0.0f : 2.0e-3f + 2.0e-4f * std::abs(reference);
    const bool mismatch = requireExact
                              ? floatBits(value) != floatBits(reference)
                              : !std::isfinite(value) || difference > tolerance;
    if (mismatch) {
      std::fprintf(stderr,
                   "M=%d N=%d K=%d output=%zu: expected %.9g, got %.9g "
                   "(difference %.9g, tolerance %.9g)\n",
                   M, N, K, index, reference, value, difference, tolerance);
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  int deviceCount = 0;
  const hipError_t countStatus = hipGetDeviceCount(&deviceCount);
  if (countStatus == hipErrorNoDevice || deviceCount == 0)
    return 77;
  if (!checkHip(countStatus, "hipGetDeviceCount"))
    return 1;

  int selectedDevice = -1;
  for (int device = 0; device < deviceCount; ++device) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) == hipSuccess &&
        (std::strcmp(properties.gcnArchName, "gfx1151") == 0 ||
         std::strncmp(properties.gcnArchName, "gfx1151:", 8) == 0)) {
      selectedDevice = device;
      break;
    }
  }
  if (selectedDevice < 0)
    return 77;
  if (!checkHip(hipSetDevice(selectedDevice), "hipSetDevice"))
    return 1;

  // The exact case's products and sums are integers F32 holds exactly. The
  // others cover both tile shapes (M < 128 and M >= 128), partial tiles in M
  // and N, a K that is not a multiple of the 64-wide slice, and a projection
  // geometry.
  return runCase(37, 77, 136, true) && runCase(5, 70, 1032, false) &&
                 runCase(200, 130, 520, false) &&
                 runCase(512, 1024, 3072, false)
             ? 0
             : 1;
}
