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
 * Algorithm: three passes over the row with shared-memory block reductions.
 *   Pass A: each block sums x over `norm_size` in FP32; thread 0 derives mean.
 *   Pass B: each block sums (x - mean)^2 in FP32; thread 0 derives inv_std.
 *   Pass C: each thread writes y in input dtype, using FP32 math.
 *
 * The variance is centered rather than E[x^2] - mean^2: for rows whose
 * variance is much smaller than mean^2 (e.g. [10000, 10001]) the latter
 * cancels to zero in FP32 and amplifies the outputs by rsqrt(epsilon). The
 * extra pass re-reads a row that is already cache-resident.
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

  __shared__ float s_mean;
  __shared__ float s_inv_std;
  __shared__ float s_reduce[kLayerNormBlockSize];

  // Pass A: mean.
  float local_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x)
    local_sum += load_as_float<T_IN, T_SCALE>(x_row + i);

  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_mean = s_reduce[0] / static_cast<float>(norm_size);
  __syncthreads();

  // Pass B: centered variance.
  float mean = s_mean;
  float local_sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float c = load_as_float<T_IN, T_SCALE>(x_row + i) - mean;
    local_sumsq += c * c;
  }

  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float var = s_reduce[0] / static_cast<float>(norm_size);
    float inv_std = rsqrtf(var + epsilon);
    s_inv_std = inv_std;
    if (mean_out)
      mean_out[row] = store_from_float<MEAN_OUT_T>(mean);
    if (inv_std_out)
      inv_std_out[row] = store_from_float<MEAN_OUT_T>(inv_std);
  }
  __syncthreads();

  // Pass C: per-element normalize + affine.
  float inv_std = s_inv_std;
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
