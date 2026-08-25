/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Softplus activation: y = log(1 + exp(x)), element-wise.
 *
 * Algorithm matches MIOpen's device path for miopenActivationSOFTRELU on the
 * packed 1D tensors hip-ep passes to miopenActivationForward:
 *   ActivFwdSolver0 -> MIOpenNeuron.cl (MIOpenActiveFwdLite)
 *   -> activation_functions.h ActivationFunction_BNLL
 *
 *   x > 0 : x + log(1 + exp(-x))
 *   x <= 0: log(1 + exp(x))
 *
 * FP16 promotes to float for exp/log (same pattern as leaky_relu_kernel.hip).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ float softplus_f32(float x) {
  return x > 0.f ? x + logf(expf(-x) + 1.f) : logf(expf(x) + 1.f);
}

__global__ void softplus_f32_kernel(const float *__restrict__ input,
                                    float *__restrict__ output,
                                    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = softplus_f32(input[idx]);
}

__global__ void softplus_f16_kernel(const __half *__restrict__ input,
                                    __half *__restrict__ output,
                                    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(softplus_f32(x));
  }
}

extern "C" int hip_softplus(void *stream, const void *input, void *output,
                            int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_softplus: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(softplus_f32_kernel, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_softplus: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(softplus_f16_kernel, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_softplus: unsupported hip_dtype=%d "
            "(only f32/f16)\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_softplus launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
