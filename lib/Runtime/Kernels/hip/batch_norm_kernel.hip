/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * BatchNormalization (inference):
 *
 *   y = scale[c] * (x - mean[c]) * rsqrt(var[c] + epsilon) + bias[c]
 *
 * Mean and variance are the supplied per-channel vectors, not reduced from
 * the input. Layout is (N, C, spatial) row-major. One block per (n, c) row.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kBatchNormBlockSize = 256;

template <typename T> struct BatchNormTraits;

template <> struct BatchNormTraits<float> {
  using Compute = float;
  __device__ static inline Compute load(float v) { return v; }
  __device__ static inline float store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct BatchNormTraits<__half> {
  using Compute = float;
  __device__ static inline Compute load(__half v) { return __half2float(v); }
  __device__ static inline __half store(Compute v) { return __float2half(v); }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct BatchNormTraits<__hip_bfloat16> {
  using Compute = float;
  __device__ static inline Compute load(__hip_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ static inline __hip_bfloat16 store(Compute v) {
    return __float2bfloat16(v);
  }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrtf(v); }
};

template <> struct BatchNormTraits<double> {
  using Compute = double;
  __device__ static inline Compute load(double v) { return v; }
  __device__ static inline double store(Compute v) { return v; }
  __device__ static inline Compute rsqrt_fn(Compute v) { return rsqrt(v); }
};

template <typename T>
__global__ void hip_batch_norm_kernel(const T *__restrict__ input,
                                      const T *__restrict__ scale,
                                      const T *__restrict__ bias,
                                      const T *__restrict__ mean,
                                      const T *__restrict__ variance,
                                      T *__restrict__ output, int64_t outer,
                                      int64_t channels, int64_t spatial,
                                      float epsilon) {
  using Tr = BatchNormTraits<T>;
  using Compute = typename Tr::Compute;

  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  int64_t channel = row % channels;
  const T *x_row = input + row * spatial;
  T *y_row = output + row * spatial;

  Compute mean_c = Tr::load(mean[channel]);
  Compute var_c = Tr::load(variance[channel]);
  Compute inv_std = Tr::rsqrt_fn(var_c + static_cast<Compute>(epsilon));
  Compute s = Tr::load(scale[channel]);
  Compute b = Tr::load(bias[channel]);
  for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
    Compute v = Tr::load(x_row[i]);
    y_row[i] = Tr::store((v - mean_c) * inv_std * s + b);
  }
}

extern "C" int hip_batch_norm(void *stream, const void *input,
                              const void *scale, const void *bias,
                              const void *mean, const void *variance,
                              void *output, int64_t n, int64_t c,
                              int64_t spatial, float epsilon, int hip_dtype) {
  if (n <= 0 || c <= 0 || spatial <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int64_t outer = n * c;
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kBatchNormBlockSize);

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_batch_norm: dtype=%d n=%lld c=%lld spatial=%lld "
      "eps=%f\n",
      hip_dtype, (long long)n, (long long)c, (long long)spatial, epsilon);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_batch_norm_kernel<__half>), grid,
                       block, 0, hip_stream, static_cast<const __half *>(input),
                       static_cast<const __half *>(scale),
                       static_cast<const __half *>(bias),
                       static_cast<const __half *>(mean),
                       static_cast<const __half *>(variance),
                       static_cast<__half *>(output), outer, c, spatial,
                       epsilon);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_batch_norm_kernel<__hip_bfloat16>), grid, block, 0,
        hip_stream, static_cast<const __hip_bfloat16 *>(input),
        static_cast<const __hip_bfloat16 *>(scale),
        static_cast<const __hip_bfloat16 *>(bias),
        static_cast<const __hip_bfloat16 *>(mean),
        static_cast<const __hip_bfloat16 *>(variance),
        static_cast<__hip_bfloat16 *>(output), outer, c, spatial, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_batch_norm_kernel<float>), grid,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<const float *>(bias),
                       static_cast<const float *>(mean),
                       static_cast<const float *>(variance),
                       static_cast<float *>(output), outer, c, spatial,
                       epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_batch_norm_kernel<double>), grid,
                       block, 0, hip_stream,
                       static_cast<const double *>(input),
                       static_cast<const double *>(scale),
                       static_cast<const double *>(bias),
                       static_cast<const double *>(mean),
                       static_cast<const double *>(variance),
                       static_cast<double *>(output), outer, c, spatial,
                       epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_batch_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_batch_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
