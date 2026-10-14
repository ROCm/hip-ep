/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

uint32_t floatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float floatFromBits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool checkHip(hipError_t status, const char *operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(status));
  return false;
}

bool freeDevice(void *pointer, const char *operation) {
  return pointer == nullptr || checkHip(hipFree(pointer), operation);
}

} // namespace

int main() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    return 77;

  // The two midpoint values exercise round-to-nearest-even in both
  // directions: 1 + 2^-8 rounds down, while 1 + 3*2^-8 rounds up.
  constexpr std::array<float, 8> input = {0.0f,        -0.0f,       1.0f,
                                          -2.5f,       1.00390625f, 1.01171875f,
                                          0.33333334f, -123.75f};
  constexpr std::array<uint16_t, input.size()> expectedBf16 = {
      0x0000, 0x8000, 0x3F80, 0xC020, 0x3F80, 0x3F82, 0x3EAB, 0xC2F8};
  std::array<float, input.size()> expectedRoundTrip{};
  for (size_t index = 0; index < input.size(); ++index) {
    expectedRoundTrip[index] =
        floatFromBits(static_cast<uint32_t>(expectedBf16[index]) << 16);
  }

  float *deviceInput = nullptr;
  uint16_t *deviceBf16 = nullptr;
  float *deviceOutput = nullptr;
  if (!checkHip(
          hipMalloc(reinterpret_cast<void **>(&deviceInput), sizeof(input)),
          "hipMalloc input") ||
      !checkHip(hipMalloc(reinterpret_cast<void **>(&deviceBf16),
                          sizeof(expectedBf16)),
                "hipMalloc bf16") ||
      !checkHip(hipMalloc(reinterpret_cast<void **>(&deviceOutput),
                          sizeof(expectedRoundTrip)),
                "hipMalloc output")) {
    freeDevice(deviceOutput, "hipFree output");
    freeDevice(deviceBf16, "hipFree bf16");
    freeDevice(deviceInput, "hipFree input");
    return 1;
  }

  bool passed = checkHip(hipMemcpy(deviceInput, input.data(), sizeof(input),
                                   hipMemcpyHostToDevice),
                         "hipMemcpy input");
  if (passed)
    passed = hip_cast(nullptr, deviceInput, deviceBf16, input.size(),
                      HIP_DTYPE_FLOAT32, HIP_DTYPE_BFLOAT16) == 0;
  if (passed)
    passed = hip_cast(nullptr, deviceBf16, deviceOutput, input.size(),
                      HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT32) == 0;
  if (passed)
    passed = checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");

  std::array<uint16_t, input.size()> actualBf16{};
  std::array<float, input.size()> actualRoundTrip{};
  if (passed)
    passed = checkHip(hipMemcpy(actualBf16.data(), deviceBf16,
                                sizeof(actualBf16), hipMemcpyDeviceToHost),
                      "hipMemcpy bf16");
  if (passed)
    passed = checkHip(hipMemcpy(actualRoundTrip.data(), deviceOutput,
                                sizeof(actualRoundTrip), hipMemcpyDeviceToHost),
                      "hipMemcpy output");

  for (size_t index = 0; passed && index < input.size(); ++index) {
    if (actualBf16[index] != expectedBf16[index] ||
        floatBits(actualRoundTrip[index]) !=
            floatBits(expectedRoundTrip[index])) {
      std::fprintf(stderr,
                   "BF16 cast mismatch at %zu: bits=0x%04x expected=0x%04x "
                   "round_trip=0x%08x expected=0x%08x\n",
                   index, actualBf16[index], expectedBf16[index],
                   floatBits(actualRoundTrip[index]),
                   floatBits(expectedRoundTrip[index]));
      passed = false;
    }
  }

  // Unsupported pairs must fail rather than silently leave the destination
  // untouched.
  if (hip_cast(nullptr, deviceInput, deviceBf16, input.size(),
               HIP_DTYPE_FLOAT64, HIP_DTYPE_BFLOAT16) == 0) {
    std::fprintf(stderr, "unsupported F64->BF16 cast unexpectedly succeeded\n");
    passed = false;
  }

  passed &= freeDevice(deviceOutput, "hipFree output");
  passed &= freeDevice(deviceBf16, "hipFree bf16");
  passed &= freeDevice(deviceInput, "hipFree input");
  return passed ? 0 : 1;
}
