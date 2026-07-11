/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Fused SkipSimplifiedLayerNormalization (com.microsoft) kernel.
 *
 *   input_skip_bias_sum = input + skip [+ bias]
 *   output              = RMSNorm(input_skip_bias_sum) * gamma
 *
 * This is the single-kernel alternative to the MIOpen composition
 * (miopenOpTensor(ADD) [+ ADD] + miopenT5LayerNormForward) in
 * real/skip_simplified_layer_norm.cpp. Selected at runtime by
 * HIPDNN_EP_FUSED_SKIPNORM=1 to remove the separate elementwise-add
 * dispatch(es) + their cache-flush barriers per layer.
 *
 * One block per row (num_rows blocks), kBS threads. FP16 accumulates in
 * float. `skip_sum` (= input_skip_bias_sum when requested, else scratch)
 * receives the pre-norm sum, mirroring the MIOpen path's intermediate so
 * downstream consumers see identical bytes.
 */

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "hip_custom_kernels.h"

namespace {
constexpr int kBS = 256;

template <typename T>
__global__ void skip_rms_norm_kernel(const T *__restrict__ input,
                                     const T *__restrict__ skip,
                                     const T *__restrict__ gamma,
                                     const T *__restrict__ bias,
                                     T *__restrict__ output,
                                     T *__restrict__ skip_sum, int64_t hidden,
                                     float eps) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t base = row * hidden;

  __shared__ float s_partial[kBS];
  __shared__ float s_rstd;

  // Pass 1: sum = input + skip [+ bias]; stash to skip_sum; accumulate sum of
  // squares in float.
  float local = 0.0f;
  for (int64_t i = tid; i < hidden; i += kBS) {
    float v = static_cast<float>(input[base + i]) +
              static_cast<float>(skip[base + i]);
    if (bias)
      v += static_cast<float>(bias[i]);
    skip_sum[base + i] = static_cast<T>(v);
    local += v * v;
  }

  s_partial[tid] = local;
  __syncthreads();
  for (int stride = kBS / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_partial[tid] += s_partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    float mean_sq = s_partial[0] / static_cast<float>(hidden);
    s_rstd = rsqrtf(mean_sq + eps);
  }
  __syncthreads();
  const float rstd = s_rstd;

  // Pass 2: output = (sum reloaded as T) * rstd * gamma. Reloading the fp16
  // stash matches the MIOpen path (which norms the stored intermediate).
  for (int64_t i = tid; i < hidden; i += kBS) {
    float v = static_cast<float>(skip_sum[base + i]);
    output[base + i] = static_cast<T>(v * rstd * static_cast<float>(gamma[i]));
  }
}
} // namespace

HIP_KERNEL_API int hip_skip_rms_norm(void *stream, const void *input,
                                     const void *skip, const void *gamma,
                                     const void *bias, void *output,
                                     void *skip_sum, int64_t num_rows,
                                     int64_t hidden, float epsilon,
                                     int hip_dtype) {
  if (num_rows <= 0 || hidden <= 0 || !input || !skip || !gamma || !output ||
      !skip_sum)
    return -1;

  hipStream_t s = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned>(num_rows));
  dim3 block(kBS);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    skip_rms_norm_kernel<__half><<<grid, block, 0, s>>>(
        static_cast<const __half *>(input), static_cast<const __half *>(skip),
        static_cast<const __half *>(gamma), static_cast<const __half *>(bias),
        static_cast<__half *>(output), static_cast<__half *>(skip_sum), hidden,
        epsilon);
    break;
  case HIP_DTYPE_FLOAT32:
    skip_rms_norm_kernel<float><<<grid, block, 0, s>>>(
        static_cast<const float *>(input), static_cast<const float *>(skip),
        static_cast<const float *>(gamma), static_cast<const float *>(bias),
        static_cast<float *>(output), static_cast<float *>(skip_sum), hidden,
        epsilon);
    break;
  default:
    return -1;
  }
  return 0;
}
