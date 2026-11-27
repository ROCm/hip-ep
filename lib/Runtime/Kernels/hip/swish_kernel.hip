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
__device__ __forceinline__ T
computeSwish(typename SwishTraits<T>::ComputeType x, double alpha) {
  using Traits = SwishTraits<T>;
  using ComputeType = typename Traits::ComputeType;

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
    output[idx] = computeSwish<T>(SwishTraits<T>::load(input[idx]), alpha);
}

// com.microsoft ScaledSiLU's z = s * x is a separate node there, so z is
// rounded to T before the swish sees it, as the hip.mul it replaces did.
template <typename T>
__global__ void scaledSwishKernel(const T *__restrict__ input,
                                  T *__restrict__ output, int64_t numElements,
                                  double alpha, double inputScale) {
  using Traits = SwishTraits<T>;
  using ComputeType = typename Traits::ComputeType;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    ComputeType s = Traits::load(Traits::store(static_cast<ComputeType>(inputScale)));
    ComputeType z = Traits::load(Traits::store(s * Traits::load(input[idx])));
    output[idx] = computeSwish<T>(z, alpha);
  }
}

template <typename T>
void launchSwish(hipStream_t stream, const void *input, void *output,
                 int64_t numElements, int gridSize, int blockSize, double alpha,
                 double inputScale) {
  const T *in = static_cast<const T *>(input);
  T *out = static_cast<T *>(output);
  if (inputScale == 1.0)
    hipLaunchKernelGGL(swishKernel<T>, dim3(gridSize), dim3(blockSize), 0,
                       stream, in, out, numElements, alpha);
  else
    hipLaunchKernelGGL(scaledSwishKernel<T>, dim3(gridSize), dim3(blockSize),
                       0, stream, in, out, numElements, alpha, inputScale);
}

extern "C" int hip_swish(void *stream, const void *input, void *output,
                         int64_t numElements, int hipDtype, double alpha,
                         double inputScale) {
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
    launchSwish<__half>(hipStream, input, output, numElements, gridSize,
                        kBlockSize, alpha, inputScale);
    break;
  case HIP_DTYPE_FLOAT32:
    launchSwish<float>(hipStream, input, output, numElements, gridSize,
                       kBlockSize, alpha, inputScale);
    break;
  case HIP_DTYPE_BFLOAT16:
    launchSwish<__hip_bfloat16>(hipStream, input, output, numElements,
                                gridSize, kBlockSize, alpha, inputScale);
    break;
  case HIP_DTYPE_FLOAT64:
    launchSwish<double>(hipStream, input, output, numElements, gridSize,
                        kBlockSize, alpha, inputScale);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_swish: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_swish: dtype=%d, alpha=%f, input_scale=%f, "
      "num=%lld\n",
      hipDtype, alpha, inputScale, (long long)numElements);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_swish launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
