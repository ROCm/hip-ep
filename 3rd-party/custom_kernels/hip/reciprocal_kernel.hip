/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>

// Reciprocal kernel for float32
__global__ void reciprocal_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = input[idx];
    // Division by zero handling: 1/0 = inf, 1/(-0) = -inf
    output[idx] = 1.0f / x;
  }
}

// Reciprocal kernel for float16
__global__ void reciprocal_f16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    __half x = input[idx];
    output[idx] = __half(1.0f) / x;
  }
}

// Reciprocal kernel for bfloat16
__global__ void reciprocal_bf16_kernel(
    const hip_bfloat16* __restrict__ input,
    hip_bfloat16* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    hip_bfloat16 x = input[idx];
    output[idx] = hip_bfloat16(1.0f) / x;
  }
}

extern "C" int hip_reciprocal(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int data_type) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (data_type) {
    case HIP_DTYPE_FLOAT32: {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_reciprocal: dtype=FLOAT32, "
          "num_elements=%lld, grid=%d, block=%d\n",
          (long long)num_elements, grid_size, kBlockSize);

      hipLaunchKernelGGL(reciprocal_f32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float*>(input),
                         static_cast<float*>(output),
                         num_elements);
      break;
    }
    case HIP_DTYPE_FLOAT16: {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_reciprocal: dtype=FLOAT16, "
          "num_elements=%lld, grid=%d, block=%d\n",
          (long long)num_elements, grid_size, kBlockSize);

      hipLaunchKernelGGL(reciprocal_f16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half*>(input),
                         static_cast<__half*>(output),
                         num_elements);
      break;
    }
    case HIP_DTYPE_BFLOAT16: {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_reciprocal: dtype=BFLOAT16, "
          "num_elements=%lld, grid=%d, block=%d\n",
          (long long)num_elements, grid_size, kBlockSize);

      hipLaunchKernelGGL(reciprocal_bf16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const hip_bfloat16*>(input),
                         static_cast<hip_bfloat16*>(output),
                         num_elements);
      break;
    }
    default:
      fprintf(stderr,
              "[custom_kernels] hip_reciprocal: unsupported dtype=%d\n",
              data_type);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_reciprocal launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
