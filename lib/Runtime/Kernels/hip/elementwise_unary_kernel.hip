/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Unary elementwise HIP kernels (Neg / Sign / Cos / Sin / Not / Sigmoid /
 * Tanh, and friends).
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
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// =====================================================================
// Abs: y = abs(x)
// ORT: UNARY_OP_NAME_EXPR(Abs, abs(a)) over CSILHFD (we cover f16/i32/i64/f32).
// =====================================================================

template <typename T>
__global__ void elementwise_abs_int_kernel(const T *__restrict__ input,
                                           T *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T a = input[idx];
    output[idx] = static_cast<T>((a < T(0)) ? -a : a);
  }
}

__global__ void elementwise_abs_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(input[idx]);
    output[idx] = __float2half(fabsf(a));
  }
}

__global__ void elementwise_abs_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = fabsf(input[idx]);
}

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

// =====================================================================
// Cos / Sin: y = cos(x) / sin(x)
//
// ORT functor (cu_inc/common.cuh):
//   _Cos<float>(a) = cosf(a);   _Sin<float>(a) = sinf(a);
//   _Cos<half>(a)  = hcos(a) (or float promote on old arch);
//   _Sin<half>(a)  = hsin(a) (or float promote on old arch).
//
// We do the float-promote path on FP16 unconditionally -- matches the
// existing elementwise_sqrt_f16 kernel and is bit-equivalent to ORT's
// __CUDA_ARCH__ < 530 branch.
// =====================================================================

__global__ void elementwise_cos_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(cosf(x));
  }
}

__global__ void elementwise_cos_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = cosf(input[idx]);
  }
}

// =====================================================================
// Erf: y = erf(x)
// =====================================================================

__global__ void elementwise_erf_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(erff(x));
  }
}

__global__ void
elementwise_erf_bf16_kernel(const __hip_bfloat16 *__restrict__ input,
                            __hip_bfloat16 *__restrict__ output,
                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(erff(x));
  }
}

__global__ void elementwise_erf_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = erff(input[idx]);
}

__global__ void elementwise_erf_f64_kernel(const double *__restrict__ input,
                                           double *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = erf(input[idx]);
}

// =====================================================================
// Not: y = !x  (bool input -> bool output, 1 byte each).
// ORT: SPECIALIZED_UNARY_ELEMENTWISE_IMPL(Not, bool) with OP_Not(a) = !a.
// In our world bool is encoded as uint8_t (1 byte) so we use that.
// =====================================================================

__global__ void elementwise_not_kernel(const uint8_t *__restrict__ input,
                                       uint8_t *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(!input[idx]);
  }
}

extern "C" int hip_elementwise_not(void *stream, const void *input,
                                   void *output, int64_t num_elements) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_elementwise_not: num=%lld\n",
      (long long)num_elements);
  hipLaunchKernelGGL(elementwise_not_kernel, dim3(grid_size),
                     dim3(kBlockSize), 0, hip_stream,
                     static_cast<const uint8_t *>(input),
                     static_cast<uint8_t *>(output), num_elements);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_not launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void elementwise_sin_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(sinf(x));
  }
}

__global__ void elementwise_sin_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = sinf(input[idx]);
  }
}

extern "C" int hip_elementwise_sin(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_sin: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sin_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sin: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sin_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sin: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sin launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void elementwise_ceil_f16_kernel(const __half *__restrict__ input,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(ceilf(x));
  }
}

__global__ void elementwise_ceil_f32_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = ceilf(input[idx]);
  }
}

extern "C" int hip_elementwise_ceil(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_ceil: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_ceil_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_ceil: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_ceil_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_ceil: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_ceil launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// ONNX Round: nearest integer, ties to even (nearbyintf / rintf).
__global__ void elementwise_round_f16_kernel(const __half *__restrict__ input,
                                             __half *__restrict__ output,
                                             int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(nearbyintf(x));
  }
}

__global__ void elementwise_round_f32_kernel(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = nearbyintf(input[idx]);
  }
}

extern "C" int hip_elementwise_round(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_round: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_round_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_round: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_round_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_round: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_round launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void elementwise_exp_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(expf(x));
  }
}

__global__ void elementwise_exp_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = expf(input[idx]);
  }
}

extern "C" int hip_elementwise_exp(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_exp: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_exp_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_exp: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_exp_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_exp: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_exp launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void
elementwise_sigmoid_f16_kernel(const __half *__restrict__ input,
                               __half *__restrict__ output,
                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(1.0f / (1.0f + expf(-x)));
  }
}

__global__ void
elementwise_sigmoid_bf16_kernel(const __hip_bfloat16 *__restrict__ input,
                                __hip_bfloat16 *__restrict__ output,
                                int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
  }
}

__global__ void
elementwise_sigmoid_f32_kernel(const float *__restrict__ input,
                               float *__restrict__ output,
                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

extern "C" int hip_elementwise_sigmoid(void *stream, const void *input,
                                       void *output, int64_t num_elements,
                                       int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sigmoid: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sigmoid_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_BFLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sigmoid: bf16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sigmoid_bf16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(input),
                       static_cast<__hip_bfloat16 *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_sigmoid: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_sigmoid_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sigmoid: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_sigmoid launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void elementwise_tanh_f16_kernel(const __half *__restrict__ input,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(tanhf(x));
  }
}

__global__ void
elementwise_tanh_bf16_kernel(const __hip_bfloat16 *__restrict__ input,
                             __hip_bfloat16 *__restrict__ output,
                             int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(tanhf(x));
  }
}

__global__ void elementwise_tanh_f32_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements)
    output[idx] = tanhf(input[idx]);
}

extern "C" int hip_elementwise_tanh(void *stream, const void *input,
                                    void *output, int64_t num_elements,
                                    int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_tanh: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_tanh_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_BFLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_tanh: bf16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_tanh_bf16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(input),
                       static_cast<__hip_bfloat16 *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_tanh: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_tanh_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_tanh: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_tanh launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

__global__ void elementwise_log_f16_kernel(const __half *__restrict__ input,
                                           __half *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float x = __half2float(input[idx]);
    output[idx] = __float2half(logf(x));
  }
}

__global__ void elementwise_log_f32_kernel(const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = logf(input[idx]);
  }
}

extern "C" int hip_elementwise_log(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_log: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_log_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_log: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_log_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_log: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_log launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_elementwise_cos(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_cos: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_cos_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_cos: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_cos_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_cos: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_cos launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_elementwise_erf(void *stream, const void *input,
                                   void *output, int64_t num_elements,
                                   int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_erf: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_erf_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_BFLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_erf: bf16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_erf_bf16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(input),
                       static_cast<__hip_bfloat16 *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_erf: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_erf_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_erf: f64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_erf_f64_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const double *>(input),
                       static_cast<double *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_erf: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_erf launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
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

extern "C" int hip_elementwise_abs(void *stream, const void *input,
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
        "[custom_kernels] hip_elementwise_abs: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_abs_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_abs: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_abs_f32_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_abs: i32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_abs_int_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_abs: i64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_abs_int_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_abs: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_abs launch failed: %s\n",
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
