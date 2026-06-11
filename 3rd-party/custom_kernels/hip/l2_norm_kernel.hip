/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * L2 normalization over the last axis:
 *   y = x * rsqrt(sum(x^2, last_axis) + epsilon)
 *
 * Fused replacement for the exporter-emitted q/k normalization chain in the
 * Qwen3.5 linear-attention layers. One block per row, FP32 accumulation.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kL2NormBlockSize = 256;

template <typename T>
__device__ inline float l2_load_as_float(const T *p) {
  return static_cast<float>(*p);
}
template <>
__device__ inline float l2_load_as_float<__half>(const __half *p) {
  return __half2float(*p);
}
template <typename T>
__device__ inline T l2_store_from_float(float v) {
  return static_cast<T>(v);
}
template <>
__device__ inline __half l2_store_from_float<__half>(float v) {
  return __float2half(v);
}

template <typename T>
__global__ void hip_l2_normalize_kernel(const T *__restrict__ input,
                                        T *__restrict__ output, int64_t outer,
                                        int64_t norm_size, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T *x_row = input + row * norm_size;
  T *y_row = output + row * norm_size;

  __shared__ float s_inv;
  __shared__ float s_reduce[kL2NormBlockSize];

  float local_sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = l2_load_as_float<T>(x_row + i);
    local_sumsq += v * v;
  }

  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_inv = rsqrtf(s_reduce[0] + epsilon);
  __syncthreads();

  float inv = s_inv;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = l2_load_as_float<T>(x_row + i);
    y_row[i] = l2_store_from_float<T>(v * inv);
  }
}

extern "C" int hip_l2_normalize(void *stream, const void *input, void *output,
                                int64_t outer, int64_t norm_size, float epsilon,
                                int hip_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kL2NormBlockSize);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_l2_normalize_kernel<__half>), grid,
                       block, 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), outer, norm_size,
                       epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_l2_normalize_kernel<float>), grid,
                       block, 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), outer, norm_size, epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_l2_normalize: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_l2_normalize launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
