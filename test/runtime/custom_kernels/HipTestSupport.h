/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- HipTestSupport.h - Common HIP test support --------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_RUNTIME_KERNELS_TEST_HIPTESTSUPPORT_H
#define HIP_RUNTIME_KERNELS_TEST_HIPTESTSUPPORT_H

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace hipdnn_ep::test {

inline bool checkHip(hipError_t error, const char *operation,
                     const char *testName) {
  if (error == hipSuccess)
    return true;
  std::fprintf(stderr, "%s: %s failed: %s\n", testName, operation,
               hipGetErrorString(error));
  return false;
}

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes) {
    if (hipMalloc(&pointer_, bytes) != hipSuccess)
      pointer_ = nullptr;
  }

  ~DeviceBuffer() {
    if (pointer_)
      (void)hipFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void *get() const { return pointer_; }

private:
  void *pointer_ = nullptr;
};

inline int findGfx1151Device(const char *testName) {
  int count = 0;
  const hipError_t status = hipGetDeviceCount(&count);
  if (status == hipErrorNoDevice || (status == hipSuccess && count == 0))
    return -1;
  if (!checkHip(status, "hipGetDeviceCount", testName))
    return -2;
  for (int device = 0; device < count; ++device) {
    hipDeviceProp_t properties{};
    if (!checkHip(hipGetDeviceProperties(&properties, device),
                  "hipGetDeviceProperties", testName))
      return -2;
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0 &&
        (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':'))
      return device;
  }
  return -1;
}

} // namespace hipdnn_ep::test

#endif // HIP_RUNTIME_KERNELS_TEST_HIPTESTSUPPORT_H
