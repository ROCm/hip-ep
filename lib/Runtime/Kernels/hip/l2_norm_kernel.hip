/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * LpNormalization (ONNX-22), last-axis fast path fused into ONE kernel.
 *
 *   p == 2:  y = x / sqrt(sum(x^2) over the norm axis)
 *   p == 1:  y = x / sum(|x|)  over the norm axis
 *
 * There is no epsilon/clamp: this reproduces exactly the numerics of the
 * decomposition in LpNormalizationConversion.cpp (Mul -> ReduceSum -> Sqrt
 * -> Div), which follows ORT CPU behaviour and emits NaN/Inf on an all-zero
 * row rather than the spec's zero-output. Keeping the same behaviour here
 * means the fused op is a drop-in for the decomposed chain.
 *
 * Layout: input is reshaped on the host as `[outer, norm_size]`, where
 *         `norm_size` is the size of the (trailing) normalization axis and
 *         `outer` is the product of all other dims. Only the last-axis case
 *         is routed here (contiguous per-row); other axes fall back to the
 *         decomposition. One block per row, FP32 accumulation.
 *
 * Modeled structurally on layer_norm_kernel.hip (single-block-per-row,
 * shared-memory reduction, FP32 math, dtype in {FLOAT16, FLOAT32}).
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

// One block per row. blockDim.x == kL2NormBlockSize. `p` is 1 or 2.
template <typename T>
__global__ void hip_l2_norm_kernel(const T *__restrict__ input,
                                   T *__restrict__ output, int64_t outer,
                                   int64_t norm_size, int p) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T *x_row = input + row * norm_size;
  T *y_row = output + row * norm_size;

  __shared__ float s_reduce[kL2NormBlockSize];
  __shared__ float s_norm;

  // Pass A: per-thread partial reduction (sum of x^2 for p==2, sum of |x| for
  // p==1), all in FP32.
  float local = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = l2_load_as_float<T>(x_row + i);
    local += (p == 1) ? fabsf(v) : v * v;
  }

  s_reduce[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float acc = s_reduce[0];
    s_norm = (p == 2) ? sqrtf(acc) : acc;
  }
  __syncthreads();

  // Pass B: divide each element by the norm (no clamp -> matches decomposition).
  float norm = s_norm;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = l2_load_as_float<T>(x_row + i);
    y_row[i] = l2_store_from_float<T>(v / norm);
  }
}

extern "C" int hip_l2_norm(void *stream, const void *input, void *output,
                           int64_t outer, int64_t norm_size, int p,
                           int hip_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;
  if (p != 1 && p != 2) {
    fprintf(stderr, "[custom_kernels] hip_l2_norm: unsupported p=%d\n", p);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kL2NormBlockSize);

  // Clear any error left sticky by a prior kernel so hipGetLastError() below
  // attributes a failure to THIS launch only.
  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_l2_norm: dtype=%d, p=%d, "
                           "outer=%lld, norm_size=%lld\n",
                           hip_dtype, p, (long long)outer, (long long)norm_size);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_l2_norm_kernel<__half>), grid, block,
                       0, hip_stream, static_cast<const __half *>(input),
                       static_cast<__half *>(output), outer, norm_size, p);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_l2_norm_kernel<float>), grid, block,
                       0, hip_stream, static_cast<const float *>(input),
                       static_cast<float *>(output), outer, norm_size, p);
  } else {
    fprintf(stderr, "[custom_kernels] hip_l2_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_l2_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
