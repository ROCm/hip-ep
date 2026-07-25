/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Custom GELU Kernel Implementation
//===----------------------------------------------------------------------===//
//
// This file implements GELU (Gaussian Error Linear Unit) activation using
// a custom HIP kernel rather than MIOpen library calls.
//
// Rationale for custom kernel:
//   MIOpen does not provide native GELU activation support as of ROCm 6.x.
//   The available miopenActivationForward() supports SIGMOID, RELU, TANH,
//   CLIPPED_RELU, ELU, and IDENTITY, but not GELU.
//
// Future considerations:
//   If future MIOpen versions add GELU support (miopenActivationMode_t_GELU),
//   consider migrating to the library implementation for potential hardware-
//   specific optimizations and reduced maintenance burden.
//
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

//===----------------------------------------------------------------------===//
// Type Traits for Unified GELU Implementation
//===----------------------------------------------------------------------===//

// Compute type: the precision used for GELU calculations
// f32/f16/bf16 use float, f64 uses double
template <typename T> struct GeluTraits;

template <> struct GeluTraits<float> {
  using ComputeType = float;
  using ConstType = float;
  static constexpr ConstType kSqrt2OverPi = 0.7978845608f;     // sqrt(2/pi)
  static constexpr ConstType kCoeff = 0.044715f;               // tanh approx coeff
  static constexpr ConstType kSqrt2Inv = 0.7071067811865476f; // 1/sqrt(2)

  __device__ static inline ComputeType load(float x) { return x; }
  __device__ static inline float store(ComputeType x) { return x; }
  __device__ static inline ComputeType tanh_fn(ComputeType x) { return tanhf(x); }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct GeluTraits<__half> {
  using ComputeType = float;
  using ConstType = float;
  static constexpr ConstType kSqrt2OverPi = 0.7978845608f;
  static constexpr ConstType kCoeff = 0.044715f;
  static constexpr ConstType kSqrt2Inv = 0.7071067811865476f;

  __device__ static inline ComputeType load(__half x) { return __half2float(x); }
  __device__ static inline __half store(ComputeType x) { return __float2half(x); }
  __device__ static inline ComputeType tanh_fn(ComputeType x) { return tanhf(x); }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct GeluTraits<__hip_bfloat16> {
  using ComputeType = float;
  using ConstType = float;
  static constexpr ConstType kSqrt2OverPi = 0.7978845608f;
  static constexpr ConstType kCoeff = 0.044715f;
  static constexpr ConstType kSqrt2Inv = 0.7071067811865476f;

  __device__ static inline ComputeType load(__hip_bfloat16 x) { return __bfloat162float(x); }
  __device__ static inline __hip_bfloat16 store(ComputeType x) { return __float2bfloat16(x); }
  __device__ static inline ComputeType tanh_fn(ComputeType x) { return tanhf(x); }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct GeluTraits<double> {
  using ComputeType = double;
  using ConstType = double;
  static constexpr ConstType kSqrt2OverPi = 0.7978845608028654; // sqrt(2/pi)
  static constexpr ConstType kCoeff = 0.044715;
  static constexpr ConstType kSqrt2Inv = 0.7071067811865476;

  __device__ static inline ComputeType load(double x) { return x; }
  __device__ static inline double store(ComputeType x) { return x; }
  __device__ static inline ComputeType tanh_fn(ComputeType x) { return tanh(x); }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erf(x); }
};

//===----------------------------------------------------------------------===//
// Unified GELU Device Function (shared computation logic)
//===----------------------------------------------------------------------===//

template <typename T>
__device__ __forceinline__ T compute_gelu(T input_val, int64_t approximate) {
  using Traits = GeluTraits<T>;
  using ComputeType = typename Traits::ComputeType;

  ComputeType x = Traits::load(input_val);
  ComputeType result;

  if (approximate == 1) {
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ComputeType x3 = x * x * x;
    ComputeType tanh_arg = Traits::kSqrt2OverPi * (x + Traits::kCoeff * x3);
    result = ComputeType(0.5) * x * (ComputeType(1.0) + Traits::tanh_fn(tanh_arg));
  } else {
    // GELU exact (erf): x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
    result = x * ComputeType(0.5) * (ComputeType(1.0) + Traits::erf_fn(x * Traits::kSqrt2Inv));
  }

  return Traits::store(result);
}

//===----------------------------------------------------------------------===//
// Specialized GELU Kernels (using shared device function)
//===----------------------------------------------------------------------===//

__global__ void elementwise_gelu_f32_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements,
                                            int64_t approximate) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = compute_gelu<float>(input[idx], approximate);
  }
}

__global__ void elementwise_gelu_f16_kernel(const __half *__restrict__ input,
                                            __half *__restrict__ output,
                                            int64_t num_elements,
                                            int64_t approximate) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = compute_gelu<__half>(input[idx], approximate);
  }
}

__global__ void elementwise_gelu_bf16_kernel(const __hip_bfloat16 *__restrict__ input,
                                             __hip_bfloat16 *__restrict__ output,
                                             int64_t num_elements,
                                             int64_t approximate) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = compute_gelu<__hip_bfloat16>(input[idx], approximate);
  }
}

__global__ void elementwise_gelu_f64_kernel(const double *__restrict__ input,
                                            double *__restrict__ output,
                                            int64_t num_elements,
                                            int64_t approximate) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = compute_gelu<double>(input[idx], approximate);
  }
}

extern "C" int hip_elementwise_gelu(void *stream, const void *input,
                                    void *output, int64_t num_elements,
                                    int hip_dtype, int64_t approximate) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  const char *mode = (approximate == 1) ? "tanh" : "erf";

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_gelu: f32, approximate=%s, "
        "num_elements=%lld, grid=%d, block=%d\n",
        mode, (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_gelu_f32_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float *>(input),
                         static_cast<float *>(output), num_elements,
                         approximate);
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_gelu: f16, approximate=%s, "
        "num_elements=%lld, grid=%d, block=%d\n",
        mode, (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_gelu_f16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half *>(input),
                         static_cast<__half *>(output), num_elements,
                         approximate);
    break;
  case HIP_DTYPE_BFLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_gelu: bf16, approximate=%s, "
        "num_elements=%lld, grid=%d, block=%d\n",
        mode, (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_gelu_bf16_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __hip_bfloat16 *>(input),
                         static_cast<__hip_bfloat16 *>(output), num_elements,
                         approximate);
    break;
  case HIP_DTYPE_FLOAT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_gelu: f64, approximate=%s, "
        "num_elements=%lld, grid=%d, block=%d\n",
        mode, (long long)num_elements, grid_size, kBlockSize);
    hipLaunchKernelGGL(elementwise_gelu_f64_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const double *>(input),
                         static_cast<double *>(output), num_elements,
                         approximate);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_gelu: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_gelu launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
