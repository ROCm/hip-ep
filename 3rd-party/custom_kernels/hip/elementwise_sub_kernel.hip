/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void elementwise_sub_i64_kernel(
    const int64_t* __restrict__ lhs,
    const int64_t* __restrict__ rhs,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_i32_kernel(
    const int32_t* __restrict__ lhs,
    const int32_t* __restrict__ rhs,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_f32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_f16_kernel(
    const __half* __restrict__ lhs,
    const __half* __restrict__ rhs,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(__half2float(lhs[idx]) - __half2float(rhs[idx]));
  }
}

extern "C" int hip_elementwise_sub(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
    case HIP_DTYPE_INT64:
      hipLaunchKernelGGL(elementwise_sub_i64_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(lhs),
                         static_cast<const int64_t*>(rhs),
                         static_cast<int64_t*>(output), num_elements);
      break;
    case HIP_DTYPE_INT32:
      hipLaunchKernelGGL(elementwise_sub_i32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int32_t*>(lhs),
                         static_cast<const int32_t*>(rhs),
                         static_cast<int32_t*>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT32:
      hipLaunchKernelGGL(elementwise_sub_f32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float*>(lhs),
                         static_cast<const float*>(rhs),
                         static_cast<float*>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT16:
      hipLaunchKernelGGL(elementwise_sub_f16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half*>(lhs),
                         static_cast<const __half*>(rhs),
                         static_cast<__half*>(output), num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_sub: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sub launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
