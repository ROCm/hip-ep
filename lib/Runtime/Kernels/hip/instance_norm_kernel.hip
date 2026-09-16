/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * InstanceNormalization: per-(N,C) mean/variance over spatial axes, then
 * channel-wise affine.
 *
 *   y = scale[c] * (x - mean) * rsqrt(var + epsilon) + bias[c]
 *
 * Layout: input/output are (N, C, spatial) row-major. One block per (n, c)
 * row; two-pass FP32 (or FP64 for f64) reduction, matching the LayerNorm
 * small-N kernel.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kInstanceNormBlockSize = 256;

template <typename T> struct InstanceNormTraits;

template <> struct InstanceNormTraits<float> {
  using Compute = float;
  __device__ static inline Compute load(float v) { return v; }
  __device__ static inline float store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct InstanceNormTraits<__half> {
  using Compute = float;
  __device__ static inline Compute load(__half v) { return __half2float(v); }
  __device__ static inline __half store(Compute v) { return __float2half(v); }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct InstanceNormTraits<__hip_bfloat16> {
  using Compute = float;
  __device__ static inline Compute load(__hip_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ static inline __hip_bfloat16 store(Compute v) {
    return __float2bfloat16(v);
  }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct InstanceNormTraits<double> {
  using Compute = double;
  __device__ static inline Compute load(double v) { return v; }
  __device__ static inline double store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrt(v); }
};

template <typename T>
__global__ void hip_instance_norm_kernel(const T *__restrict__ input,
                                         const T *__restrict__ scale,
                                         const T *__restrict__ bias,
                                         T *__restrict__ output, int64_t outer,
                                         int64_t channels, int64_t spatial,
                                         float epsilon) {
  using Tr = InstanceNormTraits<T>;
  using Compute = typename Tr::Compute;

  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  int64_t channel = row % channels;
  const T *x_row = input + row * spatial;
  T *y_row = output + row * spatial;

  __shared__ Compute s_sum;
  __shared__ Compute s_sumsq;
  __shared__ Compute s_mean;
  __shared__ Compute s_inv_std;
  __shared__ Compute s_reduce[kInstanceNormBlockSize];

  Compute local_sum = Compute(0);
  Compute local_sumsq = Compute(0);
  for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
    Compute v = Tr::load(x_row[i]);
    local_sum += v;
    local_sumsq += v * v;
  }

  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_sum = s_reduce[0];
  __syncthreads();

  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_sumsq = s_reduce[0];
    Compute mean = s_sum / static_cast<Compute>(spatial);
    Compute var =
        s_sumsq / static_cast<Compute>(spatial) - mean * mean;
    if (var < Compute(0))
      var = Compute(0);
    s_mean = mean;
    s_inv_std = Tr::rsqrt_fn(var + static_cast<Compute>(epsilon));
  }
  __syncthreads();

  Compute mean = s_mean;
  Compute inv_std = s_inv_std;
  Compute s = Tr::load(scale[channel]);
  Compute b = Tr::load(bias[channel]);
  for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
    Compute v = Tr::load(x_row[i]);
    y_row[i] = Tr::store((v - mean) * inv_std * s + b);
  }
}

extern "C" int hip_instance_norm(void *stream, const void *input,
                                 const void *scale, const void *bias,
                                 void *output, int64_t n, int64_t c,
                                 int64_t spatial, float epsilon,
                                 int hip_dtype) {
  if (n <= 0 || c <= 0 || spatial <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int64_t outer = n * c;
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kInstanceNormBlockSize);

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_instance_norm: dtype=%d n=%lld c=%lld spatial=%lld "
      "eps=%f\n",
      hip_dtype, (long long)n, (long long)c, (long long)spatial, epsilon);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_instance_norm_kernel<__half>), grid,
                       block, 0, hip_stream, static_cast<const __half *>(input),
                       static_cast<const __half *>(scale),
                       static_cast<const __half *>(bias),
                       static_cast<__half *>(output), outer, c, spatial,
                       epsilon);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_instance_norm_kernel<__hip_bfloat16>), grid, block,
        0, hip_stream, static_cast<const __hip_bfloat16 *>(input),
        static_cast<const __hip_bfloat16 *>(scale),
        static_cast<const __hip_bfloat16 *>(bias),
        static_cast<__hip_bfloat16 *>(output), outer, c, spatial, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_instance_norm_kernel<float>), grid,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<const float *>(bias),
                       static_cast<float *>(output), outer, c, spatial,
                       epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_instance_norm_kernel<double>), grid,
                       block, 0, hip_stream, static_cast<const double *>(input),
                       static_cast<const double *>(scale),
                       static_cast<const double *>(bias),
                       static_cast<double *>(output), outer, c, spatial,
                       epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_instance_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_instance_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
