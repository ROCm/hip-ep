/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Unary elementwise HIP kernels (Neg / Sign / Cos / Sin / Not, and friends).
 *
 * Ported from ONNX Runtime v1.22.2:
 *   onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.cu
 *   onnxruntime/core/providers/cuda/cu_inc/common.cuh (functor bodies)
 *
 * The ORT source uses a heavy UnaryElementWiseImpl<InT,OutT,FuncT> driver
 * template; here we use the simple hipLaunchKernelGGL + grid-stride pattern
 * already shared by elementwise_sqrt_kernel.hip / elementwise_reciprocal_kernel.hip.
 *
 * Type coverage: FP16 + INT32 + INT64 (vision.onnx-driven). bool kernels live
 * here too because Not is part of the same lowering family.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// =====================================================================
// Neg: y = -x
// ORT: UNARY_OP_NAME_EXPR(Neg, -a) over CSILHFD (we cover f16/i32/i64).
// =====================================================================

template <typename T>
__global__ void elementwise_neg_kernel(const T *__restrict__ input,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = -input[idx];
}

// FP16 specialization: ORT's _Neg on half just relies on operator-, which
// dispatches to half intrinsics. Use __hneg-equivalent via float promote to
// avoid relying on platform-specific half operators (matches the f32 path
// the f16 elementwise_sqrt kernel takes).
template <>
__global__ void elementwise_neg_kernel<__half>(const __half *__restrict__ input,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(-x);
  }
}

// =====================================================================
// Sign: y = signum(x). For signed integer / float: y = (T(0)<x) - (x<T(0)).
//
// ORT functor (cu_inc/common.cuh):
//   _Signum(a, signed)  = (T(0) < a) - (a < T(0));
//   _Sign<half>(a)      = _Signum(a, signed);
//
// ONNX delta from ORT: spec requires sign(NaN) = NaN for FP inputs. ORT's
// integer-style functor on FP types returns 0 for NaN (both comparisons are
// false). We add an explicit NaN check on the FP16 path to match ONNX spec.
// =====================================================================

template <typename T>
__global__ void elementwise_sign_int_kernel(const T *__restrict__ input,
                                            T *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T a = input[idx];
    output[idx] = static_cast<T>((T(0) < a) - (a < T(0)));
  }
}

__global__ void elementwise_sign_f16_kernel(const __half *__restrict__ input,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(input[idx]);
    float r;
    if (isnan(a)) {
      r = a;
    } else {
      r = (0.0f < a) - (a < 0.0f);
    }
    output[idx] = __float2half(r);
  }
}

__global__ void elementwise_sign_f32_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = input[idx];
    output[idx] = isnan(a) ? a : ((0.0f < a) - (a < 0.0f));
  }
}

extern "C" int hip_elementwise_sign(void *stream, const void *input,
                                    void *output, int64_t num_elements,
                                    int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sign: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sign_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sign: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sign_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sign: i32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sign_int_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sign: i64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sign_int_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sign: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sign launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_elementwise_neg(void *stream, const void *input,
                                   void *output, int64_t num_elements,
                                   int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_neg: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_neg_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_neg: i32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_neg_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_neg: i64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_neg_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_neg: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_neg_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_neg: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_neg launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
