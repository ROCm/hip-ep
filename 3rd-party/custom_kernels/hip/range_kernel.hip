/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// Range kernel: Each thread computes one output element.
// output[tid] = start + (tid * delta)
template <typename T>
__global__ void range_kernel(
    T start_val,
    T delta_val,
    T* __restrict__ output,
    int64_t num_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < num_elements) {
    output[tid] = start_val + static_cast<T>(tid) * delta_val;
  }
}

// Helper to compute range size on CPU
template <typename T>
int64_t compute_range_size(T start, T limit, T delta) {
  if (delta == T(0)) {
    return 0;
  }

  // Check if range would be empty
  if ((delta > T(0) && start >= limit) || (delta < T(0) && start <= limit)) {
    return 0;
  }

  // Compute: max(ceil((limit - start) / delta), 0)
  T range = limit - start;
  T count;

  if constexpr (std::is_floating_point<T>::value) {
    count = std::ceil(range / delta);
  } else {
    // Integer division with ceiling
    count = (range + delta - T(1)) / delta;
  }

  return static_cast<int64_t>(std::max(count, T(0)));
}

extern "C" int hip_range(
    void* stream,
    const void* start,
    const void* limit,
    const void* delta,
    void* output,
    int element_size_bytes) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;

  // Copy scalar inputs from device to host to compute output size
  // For float32/int32 (4 bytes)
  if (element_size_bytes == 4) {
    float start_val, limit_val, delta_val;
    hipMemcpy(&start_val, start, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&limit_val, limit, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&delta_val, delta, sizeof(float), hipMemcpyDeviceToHost);

    int64_t num_elements = compute_range_size(start_val, limit_val, delta_val);

    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_range: elem_size=4, "
            "start=%f, limit=%f, delta=%f, num_elements=%lld\n",
            start_val, limit_val, delta_val, (long long)num_elements);

    if (num_elements <= 0) return 0;

    int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(range_kernel<float>,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       start_val, delta_val,
                       static_cast<float*>(output),
                       num_elements);
  }
  // For float64/int64 (8 bytes)
  else if (element_size_bytes == 8) {
    double start_val, limit_val, delta_val;
    hipMemcpy(&start_val, start, sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&limit_val, limit, sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&delta_val, delta, sizeof(double), hipMemcpyDeviceToHost);

    int64_t num_elements = compute_range_size(start_val, limit_val, delta_val);

    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_range: elem_size=8, "
            "start=%f, limit=%f, delta=%f, num_elements=%lld\n",
            start_val, limit_val, delta_val, (long long)num_elements);

    if (num_elements <= 0) return 0;

    int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(range_kernel<double>,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       start_val, delta_val,
                       static_cast<double*>(output),
                       num_elements);
  }
  else {
    fprintf(stderr,
            "[custom_kernels] hip_range: unsupported element_size=%d\n",
            element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_range launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
