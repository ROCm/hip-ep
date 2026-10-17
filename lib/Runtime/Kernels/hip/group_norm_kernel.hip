/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft GroupNorm. One block per (n, group). Mean and variance are
 * computed in FP32 (FP64 for f64) over the group's channels and spatial axes,
 * then the per-channel affine is applied. activation 1 is SiLU.
 *
 * channels_last 0: (N, C, spatial). channels_last 1: (N, spatial, C).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kGroupNormBlockSize = 256;

template <typename T> struct GroupNormTraits;

template <> struct GroupNormTraits<float> {
  using Compute = float;
  __device__ static inline Compute load(float v) { return v; }
  __device__ static inline float store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<__half> {
  using Compute = float;
  __device__ static inline Compute load(__half v) { return __half2float(v); }
  __device__ static inline __half store(Compute v) { return __float2half(v); }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<__hip_bfloat16> {
  using Compute = float;
  __device__ static inline Compute load(__hip_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ static inline __hip_bfloat16 store(Compute v) {
    return __float2bfloat16(v);
  }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
  __device__ static inline Compute exp_fn(Compute v) { return expf(v); }
};

template <> struct GroupNormTraits<double> {
  using Compute = double;
  __device__ static inline Compute load(double v) { return v; }
  __device__ static inline double store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrt(v); }
  __device__ static inline Compute exp_fn(Compute v) { return exp(v); }
};

__device__ inline int64_t groupNormOffset(int64_t n, int64_t c, int64_t s,
                                          int64_t channels, int64_t spatial,
                                          int channelsLast) {
  if (channelsLast)
    return (n * spatial + s) * channels + c;
  return (n * channels + c) * spatial + s;
}

template <typename T>
__global__ void hip_group_norm_kernel(const T *__restrict__ input,
                                      const T *__restrict__ scale,
                                      const T *__restrict__ bias,
                                      T *__restrict__ output, int64_t channels,
                                      int64_t spatial, int64_t groups,
                                      int channelsLast, int activation,
                                      float epsilon) {
  using Tr = GroupNormTraits<T>;
  using Compute = typename Tr::Compute;

  int64_t row = blockIdx.x;
  int64_t nGroups = gridDim.x;
  if (row >= nGroups)
    return;

  int64_t n = row / groups;
  int64_t group = row - n * groups;
  int64_t channelsPerGroup = channels / groups;
  int64_t count = channelsPerGroup * spatial;

  __shared__ Compute s_sum;
  __shared__ Compute s_sumsq;
  __shared__ Compute s_mean;
  __shared__ Compute s_inv_std;
  __shared__ Compute s_reduce[kGroupNormBlockSize];

  Compute localSum = Compute(0);
  Compute localSumSq = Compute(0);
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    int64_t localC = i / spatial;
    int64_t s = i - localC * spatial;
    int64_t c = group * channelsPerGroup + localC;
    int64_t offset =
        groupNormOffset(n, c, s, channels, spatial, channelsLast);
    Compute v = Tr::load(input[offset]);
    localSum += v;
    localSumSq += v * v;
  }

  s_reduce[threadIdx.x] = localSum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_sum = s_reduce[0];
  __syncthreads();

  s_reduce[threadIdx.x] = localSumSq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_sumsq = s_reduce[0];
    Compute mean = s_sum / static_cast<Compute>(count);
    Compute var = s_sumsq / static_cast<Compute>(count) - mean * mean;
    if (var < Compute(0))
      var = Compute(0);
    s_mean = mean;
    s_inv_std = Tr::rsqrt_fn(var + static_cast<Compute>(epsilon));
  }
  __syncthreads();

  Compute mean = s_mean;
  Compute invStd = s_inv_std;
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    int64_t localC = i / spatial;
    int64_t s = i - localC * spatial;
    int64_t c = group * channelsPerGroup + localC;
    int64_t offset =
        groupNormOffset(n, c, s, channels, spatial, channelsLast);
    Compute v = Tr::load(input[offset]);
    Compute y = (v - mean) * invStd * Tr::load(scale[c]) + Tr::load(bias[c]);
    if (activation) {
      Compute sig = Compute(1) / (Compute(1) + Tr::exp_fn(-y));
      y = y * sig;
    }
    output[offset] = Tr::store(y);
  }
}

extern "C" int hip_group_norm(void *stream, const void *input,
                              const void *scale, const void *bias, void *output,
                              int64_t n, int64_t c, int64_t spatial,
                              int64_t groups, int channels_last, int activation,
                              float epsilon, int hip_dtype) {
  if (n <= 0 || c <= 0 || spatial <= 0)
    return 0;
  if (groups <= 0 || c % groups != 0 ||
      (channels_last != 0 && channels_last != 1) ||
      (activation != 0 && activation != 1))
    return -1;
  if (n > static_cast<int64_t>(UINT32_MAX) / groups)
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int64_t outer = n * groups;
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kGroupNormBlockSize);

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_group_norm: dtype=%d n=%lld c=%lld spatial=%lld "
      "groups=%lld channels_last=%d activation=%d eps=%f\n",
      hip_dtype, (long long)n, (long long)c, (long long)spatial,
      (long long)groups, channels_last, activation, epsilon);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_group_norm_kernel<__half>), grid,
                       block, 0, hip_stream, static_cast<const __half *>(input),
                       static_cast<const __half *>(scale),
                       static_cast<const __half *>(bias),
                       static_cast<__half *>(output), c, spatial, groups,
                       channels_last, activation, epsilon);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_group_norm_kernel<__hip_bfloat16>), grid, block, 0,
        hip_stream, static_cast<const __hip_bfloat16 *>(input),
        static_cast<const __hip_bfloat16 *>(scale),
        static_cast<const __hip_bfloat16 *>(bias),
        static_cast<__hip_bfloat16 *>(output), c, spatial, groups,
        channels_last, activation, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_group_norm_kernel<float>), grid,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<const float *>(bias),
                       static_cast<float *>(output), c, spatial, groups,
                       channels_last, activation, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_group_norm_kernel<double>), grid,
                       block, 0, hip_stream,
                       static_cast<const double *>(input),
                       static_cast<const double *>(scale),
                       static_cast<const double *>(bias),
                       static_cast<double *>(output), c, spatial, groups,
                       channels_last, activation, epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_group_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_group_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
