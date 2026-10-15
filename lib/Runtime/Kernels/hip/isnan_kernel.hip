/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * IsNaN: output[i] = 1 if input[i] is NaN, else 0.
 * Input is f16, bf16, f32, or f64. Output is one byte per element.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T> struct IsNaN;

template <> struct IsNaN<float> {
  __device__ static bool test(float value) { return isnan(value); }
};

template <> struct IsNaN<double> {
  __device__ static bool test(double value) { return isnan(value); }
};

template <> struct IsNaN<__half> {
  __device__ static bool test(__half value) {
    return isnan(__half2float(value));
  }
};

template <> struct IsNaN<__hip_bfloat16> {
  __device__ static bool test(__hip_bfloat16 value) {
    return isnan(__bfloat162float(value));
  }
};

template <typename T>
__global__ void isnan_kernel(const T *__restrict__ input,
                             uint8_t *__restrict__ output,
                             int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = static_cast<uint8_t>(IsNaN<T>::test(input[idx]));
}

template <typename T>
static int launch_isnan(hipStream_t stream, const void *input, void *output,
                        int64_t num_elements, int grid_size, int block_size) {
  hipLaunchKernelGGL(isnan_kernel<T>, dim3(grid_size), dim3(block_size), 0,
                     stream, static_cast<const T *>(input),
                     static_cast<uint8_t *>(output), num_elements);
  return 0;
}

extern "C" int hip_isnan(void *stream, const void *input, void *output,
                         int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_isnan: dtype=%d num=%lld\n",
                           hip_dtype, (long long)num_elements);

  if (hip_dtype == HIP_DTYPE_FLOAT16)
    launch_isnan<__half>(hip_stream, input, output, num_elements, grid_size,
                         kBlockSize);
  else if (hip_dtype == HIP_DTYPE_BFLOAT16)
    launch_isnan<__hip_bfloat16>(hip_stream, input, output, num_elements,
                                 grid_size, kBlockSize);
  else if (hip_dtype == HIP_DTYPE_FLOAT32)
    launch_isnan<float>(hip_stream, input, output, num_elements, grid_size,
                        kBlockSize);
  else if (hip_dtype == HIP_DTYPE_FLOAT64)
    launch_isnan<double>(hip_stream, input, output, num_elements, grid_size,
                         kBlockSize);
  else {
    fprintf(stderr, "[custom_kernels] hip_isnan: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_isnan launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
