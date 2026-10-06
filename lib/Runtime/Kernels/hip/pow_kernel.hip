/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T> struct PowTraits;

template <> struct PowTraits<float> {
  using ComputeType = float;
  __device__ static ComputeType load(float value) { return value; }
  __device__ static float store(ComputeType value) { return value; }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return powf(base, exponent);
  }
};

template <> struct PowTraits<__half> {
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

template <> struct PowTraits<__hip_bfloat16> {
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

template <> struct PowTraits<double> {
  using ComputeType = double;
  __device__ static ComputeType load(double value) { return value; }
  __device__ static double store(ComputeType value) { return value; }
  __device__ static ComputeType powFn(ComputeType base, ComputeType exponent) {
    return pow(base, exponent);
  }
};

template <typename T>
__device__ __forceinline__ T computePow(T input, double exponent) {
  using Traits = PowTraits<T>;
  using ComputeType = typename Traits::ComputeType;
  ComputeType base = Traits::load(input);
  ComputeType exp = static_cast<ComputeType>(exponent);
  return Traits::store(Traits::powFn(base, exp));
}

template <typename T>
__global__ void powKernel(const T *__restrict__ input, T *__restrict__ output,
                          int64_t numElements, double exponent) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = computePow<T>(input[idx], exponent);
}

extern "C" int hip_pow(void *stream, const void *input, void *output,
                       int64_t numElements, int hipDtype, double exponent) {
  if (numElements <= 0)
    return 0;

  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int gridSize =
      static_cast<int>((numElements + kBlockSize - 1) / kBlockSize);

  // Clear a stale launch error so the status below belongs to this kernel.
  (void)hipGetLastError();
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(powKernel<__half>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const __half *>(input),
                       static_cast<__half *>(output), numElements, exponent);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(powKernel<float>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const float *>(input),
                       static_cast<float *>(output), numElements, exponent);
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(
        powKernel<__hip_bfloat16>, dim3(gridSize), dim3(kBlockSize), 0,
        hipStream, static_cast<const __hip_bfloat16 *>(input),
        static_cast<__hip_bfloat16 *>(output), numElements, exponent);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(powKernel<double>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const double *>(input),
                       static_cast<double *>(output), numElements, exponent);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_pow: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pow: dtype=%d, exponent=%f, num=%lld\n", hipDtype,
      exponent, (long long)numElements);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_pow launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
