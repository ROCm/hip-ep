/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T> struct SwigluTraits;

template <> struct SwigluTraits<float> {
  using ComputeType = float;
  __device__ static ComputeType load(float value) { return value; }
  __device__ static float store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<__half> {
  using ComputeType = float;
  __device__ static ComputeType load(__half value) {
    return __half2float(value);
  }
  __device__ static __half store(ComputeType value) {
    return __float2half(value);
  }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static ComputeType load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __hip_bfloat16 store(ComputeType value) {
    return __float2bfloat16(value);
  }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<double> {
  using ComputeType = double;
  __device__ static ComputeType load(double value) { return value; }
  __device__ static double store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) { return exp(value); }
};

template <typename T>
__device__ __forceinline__ T computeSwiglu(T gateValue, T upValue) {
  using Traits = SwigluTraits<T>;
  using ComputeType = typename Traits::ComputeType;

  ComputeType gate = Traits::load(gateValue);
  ComputeType up = Traits::load(upValue);
  ComputeType sigmoid =
      ComputeType(1) / (ComputeType(1) + Traits::expFn(-gate));

  // Match the exported Sigmoid -> Mul -> Mul graph. Each ONNX node produces
  // a tensor of T, so round at both intermediate tensor boundaries.
  sigmoid = Traits::load(Traits::store(sigmoid));
  ComputeType activated = Traits::load(Traits::store(gate * sigmoid));
  return Traits::store(activated * up);
}

template <typename T>
__global__ void swigluKernel(const T *__restrict__ gate,
                             const T *__restrict__ up, T *__restrict__ output,
                             int64_t numElements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = computeSwiglu<T>(gate[idx], up[idx]);
}

extern "C" int hip_swiglu(void *stream, const void *gate, const void *up,
                          void *output, int64_t numElements, int hipDtype) {
  if (numElements <= 0)
    return 0;

  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int gridSize = static_cast<int>((numElements + kBlockSize - 1) / kBlockSize);

  // Clear a stale launch error so the returned status belongs to this launch.
  (void)hipGetLastError();
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(swigluKernel<__half>, dim3(gridSize), dim3(kBlockSize),
                       0, hipStream, static_cast<const __half *>(gate),
                       static_cast<const __half *>(up),
                       static_cast<__half *>(output), numElements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(swigluKernel<float>, dim3(gridSize), dim3(kBlockSize), 0,
                       hipStream, static_cast<const float *>(gate),
                       static_cast<const float *>(up),
                       static_cast<float *>(output), numElements);
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(swigluKernel<__hip_bfloat16>, dim3(gridSize),
                       dim3(kBlockSize), 0, hipStream,
                       static_cast<const __hip_bfloat16 *>(gate),
                       static_cast<const __hip_bfloat16 *>(up),
                       static_cast<__hip_bfloat16 *>(output), numElements);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(swigluKernel<double>, dim3(gridSize), dim3(kBlockSize),
                       0, hipStream, static_cast<const double *>(gate),
                       static_cast<const double *>(up),
                       static_cast<double *>(output), numElements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_swiglu: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_swiglu: dtype=%d, num=%lld\n",
                           hipDtype, (long long)numElements);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_swiglu launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
