/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * SiLU / swish activation: y = x * sigmoid(x). fp32 math, fp16/fp32 I/O.
 * Fused replacement for the exporter's separate onnx.Sigmoid + onnx.Mul.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T>
__global__ void hip_silu_kernel(const T *__restrict__ in, T *__restrict__ out,
                                int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = static_cast<float>(in[i]);
  float y = x / (1.0f + __expf(-x));
  out[i] = static_cast<T>(y);
}

template <>
__global__ void hip_silu_kernel<__half>(const __half *__restrict__ in,
                                        __half *__restrict__ out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = __half2float(in[i]);
  float y = x / (1.0f + __expf(-x));
  out[i] = __float2half(y);
}

extern "C" int hip_silu(void *stream, const void *input, void *output,
                        int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  hipStream_t s = static_cast<hipStream_t>(stream);
  constexpr int kBlock = 256;
  int64_t grid = (num_elements + kBlock - 1) / kBlock;

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_silu_kernel<__half>),
                       dim3((unsigned)grid), dim3(kBlock), 0, s,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_silu_kernel<float>),
                       dim3((unsigned)grid), dim3(kBlock), 0, s,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
  } else {
    fprintf(stderr, "[custom_kernels] hip_silu: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_silu launch failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
