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

bool runCase(int N, int K, bool requireExact) {
  std::vector<__hip_bfloat16> input(static_cast<size_t>(K));
  std::vector<__hip_bfloat16> weights(static_cast<size_t>(N) * K);
  std::vector<float> expected(static_cast<size_t>(N));
  std::vector<float> actual(static_cast<size_t>(N));

  for (size_t k = 0; k < input.size(); ++k) {
    const float value = requireExact ? 1.0f : inputValue(k);
    input[k] = __float2bfloat16(value);
  }
  for (int n = 0; n < N; ++n) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float value = requireExact ? static_cast<float>((n % 7) - 3)
                                       : weightValue(static_cast<size_t>(n), k);
      const __hip_bfloat16 rounded = __float2bfloat16(value);
      weights[static_cast<size_t>(n) * K + k] = rounded;
      sum = std::fma(static_cast<float>(input[static_cast<size_t>(k)]),
                     static_cast<float>(rounded), sum);
    }
    expected[static_cast<size_t>(n)] = sum;
  }

  Stream stream;
  DeviceBuffer<__hip_bfloat16> deviceInput(input.size());
  DeviceBuffer<__hip_bfloat16> deviceWeights(weights.size());
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

  const auto *misalignedInput = reinterpret_cast<const void *>(
      reinterpret_cast<uintptr_t>(deviceInput.get()) + sizeof(__hip_bfloat16));
  const auto *misalignedWeights = reinterpret_cast<const void *>(
      reinterpret_cast<uintptr_t>(deviceWeights.get()) +
      sizeof(__hip_bfloat16));
  if (hip_gemv_bf16_f32_transposed_b(stream.get(), misalignedInput,
                                     deviceWeights.get(), deviceOutput.get(), N,
                                     K) != -1 ||
      hip_gemv_bf16_f32_transposed_b(stream.get(), deviceInput.get(),
                                     misalignedWeights, deviceOutput.get(), N,
                                     K) != -1) {
    std::fprintf(stderr,
                 "hip_gemv_bf16_f32_transposed_b accepted unaligned input\n");
    return false;
  }

  if (hip_gemv_bf16_f32_transposed_b(stream.get(), deviceInput.get(),
                                     deviceWeights.get(), deviceOutput.get(), N,
                                     K) != 0) {
    std::fprintf(stderr, "hip_gemv_bf16_f32_transposed_b launch failed\n");
    return false;
  }
  if (!checkHip(hipMemcpyAsync(actual.data(), deviceOutput.get(),
                               actual.size() * sizeof(actual.front()),
                               hipMemcpyDeviceToHost, stream.get()),
                "hipMemcpyAsync(output)") ||
      !checkHip(hipStreamSynchronize(stream.get()), "hipStreamSynchronize"))
    return false;

  for (int n = 0; n < N; ++n) {
    const float reference = expected[static_cast<size_t>(n)];
    const float value = actual[static_cast<size_t>(n)];
    const float difference = std::abs(value - reference);
    const float tolerance =
        requireExact ? 0.0f : 2.0e-3f + 2.0e-4f * std::abs(reference);
    const bool mismatch = requireExact
                              ? floatBits(value) != floatBits(reference)
                              : !std::isfinite(value) || difference > tolerance;
    if (mismatch) {
      std::fprintf(stderr,
                   "N=%d K=%d output=%d: expected %.9g, got %.9g "
                   "(difference %.9g, tolerance %.9g)\n",
                   N, K, n, reference, value, difference, tolerance);
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

  // N=37 checks a partial final block; the second case is a real projection
  // geometry. The exact case covers FP32 output bit patterns with values whose
  // products and reductions are exactly representable.
  return runCase(37, 256, true) && runCase(1024, 3072, false) ? 0 : 1;
}
