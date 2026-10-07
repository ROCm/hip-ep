/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// ONNX LRN across channels. For each X[n, c, spatial], square_sum covers
// i in [max(0, c - floor((size-1)/2)), min(C-1, c + ceil((size-1)/2))].
// Y = X / (bias + alpha / size * square_sum) ^ beta.

template <typename T> struct LrnTraits;

template <> struct LrnTraits<float> {
  using ComputeType = float;
  __device__ static ComputeType load(float value) { return value; }
  __device__ static float store(ComputeType value) { return value; }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return powf(base, exponent);
  }
};

template <> struct LrnTraits<__half> {
  using ComputeType = float;
  __device__ static ComputeType load(__half value) {
    return __half2float(value);
  }
  __device__ static __half store(ComputeType value) {
    return __float2half(value);
  }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return powf(base, exponent);
  }
};

template <> struct LrnTraits<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static ComputeType load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __hip_bfloat16 store(ComputeType value) {
    return __float2bfloat16(value);
  }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return powf(base, exponent);
  }
};

template <> struct LrnTraits<double> {
  using ComputeType = double;
  __device__ static ComputeType load(double value) { return value; }
  __device__ static double store(ComputeType value) { return value; }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return pow(base, exponent);
  }
};

template <typename T>
__global__ void lrn_kernel(const T *__restrict__ input, T *__restrict__ output,
                           int64_t n, int64_t channels, int64_t spatial,
                           int64_t size, double alpha, double beta,
                           double bias) {
  using Traits = LrnTraits<T>;
  using ComputeType = typename Traits::ComputeType;
  const ComputeType alphaValue = static_cast<ComputeType>(alpha);
  const ComputeType betaValue = static_cast<ComputeType>(beta);
  const ComputeType biasValue = static_cast<ComputeType>(bias);
  const ComputeType sizeValue = static_cast<ComputeType>(size);
  const int64_t count = n * channels * spatial;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += stride) {
    const int64_t hw = idx % spatial;
    const int64_t nc = idx / spatial;
    const int64_t channel = nc % channels;
    const int64_t batch = nc / channels;

    int64_t begin = channel - (size - 1) / 2;
    if (begin < 0)
      begin = 0;
    int64_t end = channel + size / 2;
    if (end >= channels)
      end = channels - 1;

    const T *batchBase = input + (batch * channels) * spatial + hw;
    ComputeType squareSum = 0;
    for (int64_t i = begin; i <= end; ++i) {
      ComputeType value = Traits::load(batchBase[i * spatial]);
      squareSum += value * value;
    }

    ComputeType x = Traits::load(batchBase[channel * spatial]);
    ComputeType denom = biasValue + (alphaValue / sizeValue) * squareSum;
    output[idx] = Traits::store(x / Traits::powFn(denom, betaValue));
  }
}

extern "C" int hip_lrn(void *stream, const void *input, void *output, int64_t n,
                       int64_t channels, int64_t spatial, int64_t size,
                       double alpha, double beta, double bias, int hip_dtype) {
  if (n == 0 || channels == 0 || spatial == 0)
    return 0;
  if (n < 0 || channels < 0 || spatial < 0 || size < 1)
    return -1;

  const int64_t count = n * channels * spatial;
  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int64_t blocks = (count + kBlockSize - 1) / kBlockSize;
  if (blocks > 65535)
    blocks = 65535;

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(lrn_kernel<float>, dim3(static_cast<unsigned>(blocks)),
                       dim3(kBlockSize), 0, hipStream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), n, channels, spatial, size,
                       alpha, beta, bias);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(lrn_kernel<__half>, dim3(static_cast<unsigned>(blocks)),
                       dim3(kBlockSize), 0, hipStream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), n, channels, spatial,
                       size, alpha, beta, bias);
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(
        lrn_kernel<__hip_bfloat16>, dim3(static_cast<unsigned>(blocks)),
        dim3(kBlockSize), 0, hipStream,
        static_cast<const __hip_bfloat16 *>(input),
        static_cast<__hip_bfloat16 *>(output), n, channels, spatial, size,
        alpha, beta, bias);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(lrn_kernel<double>, dim3(static_cast<unsigned>(blocks)),
                       dim3(kBlockSize), 0, hipStream,
                       static_cast<const double *>(input),
                       static_cast<double *>(output), n, channels, spatial,
                       size, alpha, beta, bias);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_lrn: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_lrn launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
