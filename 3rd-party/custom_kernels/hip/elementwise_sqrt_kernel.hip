/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void elementwise_sqrt_f32_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = sqrtf(input[idx]);
}

__global__ void elementwise_sqrt_f16_kernel(const __half *__restrict__ input,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(sqrtf(x));
  }
}

__global__ void elementwise_sqrt_bf16_kernel(
    const __hip_bfloat16 *__restrict__ input,
    __hip_bfloat16 *__restrict__ output, int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(sqrtf(x));
  }
}

extern "C" int hip_elementwise_sqrt(void *stream, const void *input,
                                    void *output, int64_t num_elements,
                                    int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sqrt: f32, "
        "num_elements=%lld, grid=%d, block=%d\n",
        (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_sqrt_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sqrt: f16, "
        "num_elements=%lld, grid=%d, block=%d\n",
        (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_sqrt_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_BFLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sqrt: bf16, "
        "num_elements=%lld, grid=%d, block=%d\n",
        (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_sqrt_bf16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(input),
                       static_cast<__hip_bfloat16 *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sqrt: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sqrt launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
