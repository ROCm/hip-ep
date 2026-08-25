/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>

#include "gpu_touch.hpp"

namespace {

__global__ void BumpBf16Kernel(uint16_t* data, size_t n, uint16_t bits_delta) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = static_cast<uint16_t>(data[i] + bits_delta);
  }
}

}  // namespace

extern "C" bool GpuBumpBf16InPlace(void* host_ptr, size_t n_elems,
                                   uint16_t bits_delta) {
  void* dev_ptr = nullptr;
  if (hipHostGetDevicePointer(&dev_ptr, host_ptr, 0) != hipSuccess) {
    return false;
  }

  constexpr int kBlock = 64;
  const int grid = static_cast<int>((n_elems + kBlock - 1) / kBlock);

  (void)hipGetLastError();  // clear stale state before the launch
  BumpBf16Kernel<<<grid, kBlock>>>(static_cast<uint16_t*>(dev_ptr), n_elems,
                                   bits_delta);
  if (hipGetLastError() != hipSuccess) {
    return false;
  }
  return hipDeviceSynchronize() == hipSuccess;
}
