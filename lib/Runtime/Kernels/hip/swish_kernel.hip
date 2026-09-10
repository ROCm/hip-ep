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

template <typename T> struct SwishTraits;

template <> struct SwishTraits<float> {
  using ComputeType = float;
  __device__ static ComputeType load(float value) { return value; }
  __device__ static float store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) {
    return expf(value);
  }
};

template <> struct SwishTraits<__half> {
  using ComputeType = float;
  __device__ static ComputeType load(__half value) {
    return __half2float(value);
  }
  __device__ static __half store(ComputeType value) {
    return __float2half(value);
  }
  __device__ static ComputeType expFn(ComputeType value) {
    return expf(value);
  }
};

template <> struct SwishTraits<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static ComputeType load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __hip_bfloat16 store(ComputeType value) {
    return __float2bfloat16(value);
  }
  __device__ static ComputeType expFn(ComputeType value) {
    return expf(value);
  }
};

template <> struct SwishTraits<double> {
  using ComputeType = double;
  __device__ static ComputeType load(double value) { return value; }
  __device__ static double store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) {
    return exp(value);
  }
};

template <typename T>
__device__ __forceinline__ T computeSwish(T input, double alpha) {
  using Traits = SwishTraits<T>;
  using ComputeType = typename Traits::ComputeType;

  ComputeType x = Traits::load(input);

  // ONNX defines Swish as a function of CastLike, Mul, Sigmoid, and Mul.
  // Round each intermediate back to T so low-precision behavior matches that
  // function body rather than an unconstrained higher-precision fusion.
  ComputeType typedAlpha =
      Traits::load(Traits::store(static_cast<ComputeType>(alpha)));
  ComputeType scaled = Traits::load(Traits::store(typedAlpha * x));
  ComputeType sigmoid = ComputeType(1) /
                        (ComputeType(1) + Traits::expFn(-scaled));
  sigmoid = Traits::load(Traits::store(sigmoid));
  return Traits::store(x * sigmoid);
}

template <typename T>
__global__ void swishKernel(const T *__restrict__ input,
                            T *__restrict__ output, int64_t numElements,
                            double alpha) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = computeSwish<T>(input[idx], alpha);
}

extern "C" int hip_swish(void *stream, const void *input, void *output,
                         int64_t numElements, int hipDtype, double alpha) {
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
    hipLaunchKernelGGL(swishKernel<__half>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const __half *>(input),
                       static_cast<__half *>(output), numElements, alpha);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(swishKernel<float>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const float *>(input),
                       static_cast<float *>(output), numElements, alpha);
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(
        swishKernel<__hip_bfloat16>, dim3(gridSize), dim3(kBlockSize), 0,
        hipStream, static_cast<const __hip_bfloat16 *>(input),
        static_cast<__hip_bfloat16 *>(output), numElements, alpha);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(swishKernel<double>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const double *>(input),
                       static_cast<double *>(output), numElements, alpha);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_swish: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_swish: dtype=%d, alpha=%f, num=%lld\n", hipDtype,
      alpha, (long long)numElements);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_swish launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
