/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename Tin, typename Tout>
__global__ void cast_kernel_t(const Tin *__restrict__ input,
                               Tout *__restrict__ output,
                               int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<Tout>(input[idx]);
  }
}

// Specialisations for f16/bf16 conversions where static_cast doesn't
// compile cleanly across (vendor) HIP toolchains.
__global__ void cast_f32_to_bf16_kernel(const float *__restrict__ input,
                                         __hip_bfloat16 *__restrict__ output,
                                         int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __float2bfloat16(input[idx]);
}

__global__ void cast_bf16_to_f32_kernel(const __hip_bfloat16 *__restrict__ input,
                                         float *__restrict__ output,
                                         int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __bfloat162float(input[idx]);
}

__global__ void cast_f32_to_f16_kernel(const float *__restrict__ input,
                                        __half *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __float2half(input[idx]);
}

__global__ void cast_f16_to_f32_kernel(const __half *__restrict__ input,
                                        float *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = __half2float(input[idx]);
}

__global__ void cast_i64_to_i32_kernel(
    const int64_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
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

#define CAST_LAUNCH(_K, _Tin, _Tout)                                            \
  hipLaunchKernelGGL((_K), dim3(grid_size), dim3(kBlockSize), 0, hip_stream,    \
                     static_cast<const _Tin *>(input),                          \
                     static_cast<_Tout *>(output), num_elements)

  if (input_dtype == HIP_DTYPE_INT64 && output_dtype == HIP_DTYPE_INT32) {
    CAST_LAUNCH(cast_i64_to_i32_kernel, int64_t, int32_t);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_INT64) {
    CAST_LAUNCH((cast_kernel_t<float, int64_t>), float, int64_t);
  } else if (input_dtype == HIP_DTYPE_INT64 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH((cast_kernel_t<int64_t, float>), int64_t, float);
  } else if (input_dtype == HIP_DTYPE_INT32 &&
             output_dtype == HIP_DTYPE_INT64) {
    CAST_LAUNCH((cast_kernel_t<int32_t, int64_t>), int32_t, int64_t);
  } else if (input_dtype == HIP_DTYPE_INT32 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH((cast_kernel_t<int32_t, float>), int32_t, float);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_INT32) {
    CAST_LAUNCH((cast_kernel_t<float, int32_t>), float, int32_t);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_BFLOAT16) {
    CAST_LAUNCH(cast_f32_to_bf16_kernel, float, __hip_bfloat16);
  } else if (input_dtype == HIP_DTYPE_BFLOAT16 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_bf16_to_f32_kernel, __hip_bfloat16, float);
  } else if (input_dtype == HIP_DTYPE_FLOAT32 &&
             output_dtype == HIP_DTYPE_FLOAT16) {
    CAST_LAUNCH(cast_f32_to_f16_kernel, float, __half);
  } else if (input_dtype == HIP_DTYPE_FLOAT16 &&
             output_dtype == HIP_DTYPE_FLOAT32) {
    CAST_LAUNCH(cast_f16_to_f32_kernel, __half, float);
  } else if (input_dtype == output_dtype) {
    // Identity: bytewise copy.
    size_t elem_size = 0;
    switch (input_dtype) {
    case HIP_DTYPE_FLOAT32:
    case HIP_DTYPE_INT32:
      elem_size = 4; break;
    case HIP_DTYPE_FLOAT16:
    case HIP_DTYPE_BFLOAT16:
      elem_size = 2; break;
    case HIP_DTYPE_INT64:
      elem_size = 8; break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_cast identity: unsupported dtype=%d\n",
              input_dtype);
      return -1;
    }
    return static_cast<int>(hipMemcpyAsync(output, input,
                                            num_elements * elem_size,
                                            hipMemcpyDeviceToDevice,
                                            hip_stream));
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_cast: unsupported conversion "
            "input_dtype=%d -> output_dtype=%d\n",
            input_dtype, output_dtype);
    return -1;
  }
#undef CAST_LAUNCH

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_cast launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
