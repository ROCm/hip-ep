/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

__global__ void cast_i64_to_i32_kernel(
    const int64_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_f32_to_f16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void cast_f16_to_f32_kernel(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __half2float(input[idx]);
  }
}

__global__ void cast_i64_to_f32_kernel(
    const int64_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_i32_to_f32_kernel(
    const int32_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void cast_i64_to_f16_kernel(
    const int64_t* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(static_cast<float>(input[idx]));
  }
}

__global__ void cast_i32_to_f16_kernel(
    const int32_t* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(static_cast<float>(input[idx]));
  }
}

__global__ void cast_i32_to_i64_kernel(
    const int32_t* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_f32_to_i64_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

__global__ void cast_f32_to_i32_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void cast_f16_to_i64_kernel(
    const __half* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int64_t>(__half2float(input[idx]));
  }
}

__global__ void cast_f16_to_i32_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(__half2float(input[idx]));
  }
}

extern "C" int hip_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_INT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: INT64->INT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_i64_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<int32_t*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_FLOAT16) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: FLOAT32->FLOAT16, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_f32_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<__half*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_FLOAT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: FLOAT16->FLOAT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(cast_f16_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<float*>(output),
                       num_elements);
  } else if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_FLOAT32) {
    CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_cast: INT64->FLOAT32, num_elements=%lld\n", (long long)num_elements);
    hipLaunchKernelGGL(cast_i64_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<float*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(cast_i32_to_f32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<float*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(cast_i64_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<__half*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(cast_i32_to_f16_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<__half*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_INT32 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_i32_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_f32_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 && output_dtype == HIP_DTYPE_INT32) {
    hipLaunchKernelGGL(cast_f32_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float*>(input),
                       static_cast<int32_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_INT64) {
    hipLaunchKernelGGL(cast_f16_to_i64_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<int64_t*>(output), num_elements);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_INT32) {
    hipLaunchKernelGGL(cast_f16_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half*>(input),
                       static_cast<int32_t*>(output), num_elements);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_cast: unsupported conversion "
            "input_dtype=%d -> output_dtype=%d\n",
            input_dtype, output_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_cast launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
