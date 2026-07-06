/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Unary activations (sigmoid / relu / tanh / softplus) as custom HIP kernels.
//
// MIOpen's miopenActivationForward is high-overhead on gfx1151 for the tiny
// decode tensors these run on (shared-expert SwiGLU sigmoid, linear-attention
// tanh/softplus gating -- ~180 calls/token at ~0 GB/s = launch/dispatch bound).
// A plain elementwise kernel has far lower per-call overhead; this mirrors the
// existing custom gelu/leaky_relu path (elementwise_gelu_kernel.hip).
//
// mode: 0=sigmoid, 1=relu, 2=tanh, 3=softplus (see hip_activation()).
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ float act_apply(float x, int mode) {
  switch (mode) {
  case 0: // sigmoid
    return 1.0f / (1.0f + __expf(-x));
  case 1: // relu
    return x > 0.0f ? x : 0.0f;
  case 2: // tanh
    return tanhf(x);
  case 3: // softplus: log(1+exp(x)), numerically stable
    return x > 20.0f ? x : log1pf(__expf(x));
  default:
    return x;
  }
}

__global__ void activation_f32_kernel(const float *__restrict__ input,
                                      float *__restrict__ output,
                                      int64_t n, int mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = act_apply(input[idx], mode);
}

__global__ void activation_f16_kernel(const __half *__restrict__ input,
                                      __half *__restrict__ output,
                                      int64_t n, int mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
    output[idx] = __float2half(act_apply(__half2float(input[idx]), mode));
}

// mode: 0=sigmoid, 1=relu, 2=tanh, 3=softplus. Returns -2 for unsupported dtype
// (caller should fall back to MIOpen), other nonzero = launch error.
extern "C" int hip_activation(void *stream, const void *input, void *output,
                              int64_t num_elements, int hip_dtype, int mode) {
  if (num_elements <= 0)
    return 0;
  hipStream_t s = static_cast<hipStream_t>(stream);
  constexpr int BS = 256;
  int grid = static_cast<int>((num_elements + BS - 1) / BS);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(activation_f32_kernel, dim3(grid), dim3(BS), 0, s,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements, mode);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(activation_f16_kernel, dim3(grid), dim3(BS), 0, s,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements, mode);
    break;
  default:
    return -2; // caller falls back to MIOpen (e.g. bf16)
  }
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_activation launch failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
