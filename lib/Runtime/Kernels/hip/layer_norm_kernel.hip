/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * LayerNormalization (ONNX-17): per-row mean/variance normalization
 * followed by elementwise (scale, bias). Bias and the optional `mean` /
 * `inv_std` outputs may be null.
 *
 *   y = (x - mean) * rsqrt(var + epsilon) * scale + bias
 *
 * Layout: input is reshaped as `[outer, norm_size]` on the host
 *         (outer = input_num_elements / scale_num_elements,
 *          norm_size = scale_num_elements).
 *
 * Algorithm: one block per row, two passes over the row in FP32.
 *   Pass A: sum d = x - x[0] and d^2, reduced together in one block tree.
 *           mean = x[0] + E[d] and var = E[d^2] - E[d]^2. The shift keeps
 *           the sums at the scale of the spread rather than of |mean|, so
 *           the difference does not cancel the way E[x^2] - mean^2 does when
 *           |mean| >> stddev, and a row without spread has a variance of
 *           exactly 0.
 *   Pass B: write y in the input dtype.
 * One paired reduction, not a mean reduction followed by a deviation pass:
 * on rows of a few hundred to a few thousand elements the block reductions,
 * not the reads, set the kernel's time.
 *
 * Source: onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu @ v1.22.2
 *         (structurally; we keep the simpler "small N" branch -- one
 *          block per row -- and avoid the parallel-reduce-tree variant).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kLayerNormBlockSize = 256;

template <typename T_IN, typename T_SCALE>
__device__ inline float load_as_float(const T_IN *p) {
  return static_cast<float>(*p);
}

template <>
__device__ inline float load_as_float<__half, __half>(const __half *p) {
  return __half2float(*p);
}

template <typename T>
__device__ inline T store_from_float(float v) {
  return static_cast<T>(v);
}

template <>
__device__ inline __half store_from_float<__half>(float v) {
  return __float2half(v);
}

// Every thread of the block gets the sums of `v` over the block, added in a
// fixed tree order.
__device__ inline float2 block_sum2(float2 v, float2 *s_reduce) {
  s_reduce[threadIdx.x] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x].x += s_reduce[threadIdx.x + stride].x;
      s_reduce[threadIdx.x].y += s_reduce[threadIdx.x + stride].y;
    }
    __syncthreads();
  }
  return s_reduce[0];
}

// One block per row. blockDim.x == kLayerNormBlockSize.
// MEAN_OUT_T / INV_STD_OUT_T = float or __half (depending on stash_type).
template <typename T_IN, typename T_SCALE, typename MEAN_OUT_T>
__global__ void hip_layer_norm_kernel(const T_IN *__restrict__ input,
                                      const T_SCALE *__restrict__ scale,
                                      const T_SCALE *__restrict__ bias,
                                      T_IN *__restrict__ output,
                                      MEAN_OUT_T *__restrict__ mean_out,
                                      MEAN_OUT_T *__restrict__ inv_std_out,
                                      int64_t outer, int64_t norm_size,
                                      float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T_IN *x_row = input + row * norm_size;
  T_IN *y_row = output + row * norm_size;

  __shared__ float2 s_reduce[kLayerNormBlockSize];
  const float count = static_cast<float>(norm_size);

  // Pass A: sums of d = x - x[0] and d^2 in one read.
  const float shift = load_as_float<T_IN, T_SCALE>(x_row);
  float2 local = {0.0f, 0.0f};
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float d = load_as_float<T_IN, T_SCALE>(x_row + i) - shift;
    local.x += d;
    local.y += d * d;
  }
  const float2 sums = block_sum2(local, s_reduce);
  const float mean_d = sums.x / count;
  const float mean = shift + mean_d;
  // Rounding can take the difference just below 0; a NaN stays NaN.
  float var = sums.y / count - mean_d * mean_d;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + epsilon);
  if (threadIdx.x == 0) {
    if (mean_out)
      mean_out[row] = store_from_float<MEAN_OUT_T>(mean);
    if (inv_std_out)
      inv_std_out[row] = store_from_float<MEAN_OUT_T>(inv_std);
  }

  // Pass B: per-element normalize + affine.
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = load_as_float<T_IN, T_SCALE>(x_row + i);
    float s = load_as_float<T_SCALE, T_SCALE>(scale + i);
    float b = bias ? load_as_float<T_SCALE, T_SCALE>(bias + i) : 0.0f;
    float n = (v - mean) * inv_std * s + b;
    y_row[i] = store_from_float<T_IN>(n);
  }
}

extern "C" int hip_layer_norm(void *stream, const void *input,
                              const void *scale, const void *bias,
                              void *output, void *mean_out, void *inv_std_out,
                              int64_t outer, int64_t norm_size, float epsilon,
                              int hip_dtype, int mean_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kLayerNormBlockSize);

  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to THIS
  // launch only (otherwise an upstream error is misreported as
  // "hip_layer_norm launch failed").
  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_layer_norm: dtype=%d, mean_dtype=%d, "
      "outer=%lld, norm_size=%lld, epsilon=%f, "
      "mean=%s, inv_std=%s\n",
      hip_dtype, mean_dtype, (long long)outer, (long long)norm_size, epsilon,
      mean_out ? "yes" : "null", inv_std_out ? "yes" : "null");

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    if (mean_dtype == HIP_DTYPE_FLOAT16) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_layer_norm_kernel<__half, __half, __half>),
          grid, block, 0, hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(scale),
          static_cast<const __half *>(bias), static_cast<__half *>(output),
          static_cast<__half *>(mean_out),
          static_cast<__half *>(inv_std_out), outer, norm_size, epsilon);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_layer_norm_kernel<__half, __half, float>), grid,
          block, 0, hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(scale),
          static_cast<const __half *>(bias), static_cast<__half *>(output),
          static_cast<float *>(mean_out), static_cast<float *>(inv_std_out),
          outer, norm_size, epsilon);
    }
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_layer_norm_kernel<float, float, float>),
                       grid, block, 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<const float *>(bias),
                       static_cast<float *>(output),
                       static_cast<float *>(mean_out),
                       static_cast<float *>(inv_std_out), outer, norm_size,
                       epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_layer_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_layer_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
