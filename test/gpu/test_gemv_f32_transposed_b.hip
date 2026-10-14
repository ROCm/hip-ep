/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
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

float inputValue(size_t index) {
  const int value = static_cast<int>((index * 17 + 11) % 257) - 128;
  return static_cast<float>(value) / 256.0f;
}

float weightValue(size_t output, size_t index) {
  const int value =
      static_cast<int>((output * 29 + index * 13 + 7) % 251) - 125;
  return static_cast<float>(value) / 384.0f;
}

bool runCase(int N, int K) {
  std::vector<float> input(static_cast<size_t>(K));
  std::vector<float> weights(static_cast<size_t>(N) * K);
  std::vector<float> expected(static_cast<size_t>(N));
  std::vector<float> actual(static_cast<size_t>(N));

  for (size_t k = 0; k < input.size(); ++k)
    input[k] = inputValue(k);
  for (int n = 0; n < N; ++n) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float weight = weightValue(static_cast<size_t>(n), k);
      weights[static_cast<size_t>(n) * K + k] = weight;
      sum = std::fma(input[static_cast<size_t>(k)], weight, sum);
    }
    expected[static_cast<size_t>(n)] = sum;
  }

  Stream stream;
  DeviceBuffer<float> deviceInput(input.size());
  DeviceBuffer<float> deviceWeights(weights.size());
  DeviceBuffer<float> deviceOutput(actual.size());
  if (!checkHip(stream.status(), "hipStreamCreate") ||
      !checkHip(deviceInput.status(), "hipMalloc(input)") ||
      !checkHip(deviceWeights.status(), "hipMalloc(weights)") ||
      !checkHip(deviceOutput.status(), "hipMalloc(output)"))
    return false;

  if (!checkHip(hipMemcpyAsync(deviceInput.get(), input.data(),
                               input.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream.get()),
                "hipMemcpyAsync(input)") ||
      !checkHip(hipMemcpyAsync(deviceWeights.get(), weights.data(),
                               weights.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream.get()),
                "hipMemcpyAsync(weights)"))
    return false;

  if (hip_gemv_f32_transposed_b(stream.get(), deviceInput.get(),
                                deviceWeights.get(), deviceOutput.get(), N,
                                K) != 0) {
    std::fprintf(stderr, "hip_gemv_f32_transposed_b launch failed\n");
    return false;
  }
  if (!checkHip(hipMemcpyAsync(actual.data(), deviceOutput.get(),
                               actual.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream.get()),
                "hipMemcpyAsync(output)") ||
      !checkHip(hipStreamSynchronize(stream.get()), "hipStreamSynchronize"))
    return false;

  for (int n = 0; n < N; ++n) {
    const float reference = expected[static_cast<size_t>(n)];
    const float difference =
        std::abs(actual[static_cast<size_t>(n)] - reference);
    const float tolerance = 5.0e-4f + 1.0e-4f * std::abs(reference);
    if (difference > tolerance) {
      std::fprintf(stderr,
                   "N=%d K=%d output=%d: expected %.9g, got %.9g "
                   "(difference %.9g, tolerance %.9g)\n",
                   N, K, n, reference, actual[static_cast<size_t>(n)],
                   difference, tolerance);
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
  return runCase(1, 256) && runCase(128, 1024) && runCase(256, 4096) ? 0 : 1;
}
