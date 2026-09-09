/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Binary elementwise HIP kernels (Div / Mod / Equal / Less, and friends).
 *
 * Ported from ONNX Runtime v1.22.2:
 *   onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu
 *
 * Important constraint: the HipToLLVM lowerings (Div / Mod / Equal /
 * Less) pass only `num_elements` -- no per-input shapes. Broadcasting is
 * NOT performed here; both inputs must already have identical layout.
 * Broadcasting must be materialised upstream via Expand. This matches
 * the BinaryElementWiseNoBroadcastImpl fast-path in the CUDA EP.
 *
 * Type coverage (per-op):
 *   Div   : FP16 + INT64  (vision.onnx Div is FP16 + shape-arith INT64)
 *   Mod   : INT64 (python-style %), FP16 (fmod) -- per the `fmod` flag
 *   Equal : FP16 + INT64 -> bool (1 byte out)
 *   Less  : FP16 + INT64 -> bool (1 byte out)
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// =====================================================================
// Div: y = a / b
// ORT: BINARY_OP_NAME_EXPR(Div, (a / b))
// =====================================================================

template <typename T>
__global__ void elementwise_div_kernel(const T *__restrict__ lhs,
                                       const T *__restrict__ rhs,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] / rhs[idx];
  }
}

// FP16 specialization: promote to float to avoid relying on half /-operator.
template <>
__global__ void elementwise_div_kernel<__half>(const __half *__restrict__ lhs,
                                               const __half *__restrict__ rhs,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half(a / b);
  }
}

// =====================================================================
// Equal: y = (a == b)  -- bool output (uint8, 1 byte).
// Less:  y = (a <  b)  -- bool output (uint8, 1 byte).
// ORT: BINARY_OP_NAME_EXPR2(Equal, (a == b)) / (Less, (a < b))
// =====================================================================

// stride = (operand_num == 1) ? 0 : 1 -- scalar broadcast comes for free
// by reading address 0 for every output index. We pass strides instead of
// the per-input counts so the kernel can index without a branch.

template <typename T>
__global__ void elementwise_equal_kernel(const T *__restrict__ lhs,
                                         const T *__restrict__ rhs,
                                         uint8_t *__restrict__ output,
                                         int64_t num_elements,
                                         int64_t lhs_stride,
                                         int64_t rhs_stride) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(lhs[idx * lhs_stride] ==
                                       rhs[idx * rhs_stride]);
  }
}

// FP16 path: promote to float to avoid relying on half operator==.
__global__ void elementwise_equal_f16_kernel(
    const __half *__restrict__ lhs, const __half *__restrict__ rhs,
    uint8_t *__restrict__ output, int64_t num_elements,
    int64_t lhs_stride, int64_t rhs_stride) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx * lhs_stride]);
    float b = __half2float(rhs[idx * rhs_stride]);
    output[idx] = static_cast<uint8_t>(a == b);
  }
}

extern "C" int hip_elementwise_equal(void *stream, const void *lhs,
                                     const void *rhs, void *output,
                                     int64_t lhs_num_elements,
                                     int64_t rhs_num_elements,
                                     int64_t out_num_elements, int hip_dtype) {
  if (out_num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((out_num_elements + kBlockSize - 1) / kBlockSize);

  // stride 0 = scalar broadcast (every idx reads the same single element);
  // stride 1 = full-shape per-element read.
  int64_t lhs_stride = (lhs_num_elements == 1) ? 0 : 1;
  int64_t rhs_stride = (rhs_num_elements == 1) ? 0 : 1;

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_equal: f16, out=%lld lhs_stride=%lld "
        "rhs_stride=%lld\n",
        (long long)out_num_elements, (long long)lhs_stride,
        (long long)rhs_stride);
    hipLaunchKernelGGL(elementwise_equal_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<uint8_t *>(output), out_num_elements,
                       lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_equal: f32, out=%lld\n",
        (long long)out_num_elements);
    hipLaunchKernelGGL(elementwise_equal_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<uint8_t *>(output), out_num_elements,
                       lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_equal: i32, out=%lld\n",
        (long long)out_num_elements);
    hipLaunchKernelGGL(elementwise_equal_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<uint8_t *>(output), out_num_elements,
                       lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_equal: i64, out=%lld lhs_stride=%lld "
        "rhs_stride=%lld\n",
        (long long)out_num_elements, (long long)lhs_stride,
        (long long)rhs_stride);
    hipLaunchKernelGGL(elementwise_equal_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<uint8_t *>(output), out_num_elements,
                       lhs_stride, rhs_stride);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_equal: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_equal launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// =====================================================================
// Less: y = (a < b)  -- bool output (uint8, 1 byte).
// ORT: BINARY_OP_NAME_EXPR2(Less, (a < b))
// =====================================================================

template <typename T>
__global__ void elementwise_less_kernel(const T *__restrict__ lhs,
                                        const T *__restrict__ rhs,
                                        uint8_t *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(lhs[idx] < rhs[idx]);
  }
}

__global__ void elementwise_less_f16_kernel(const __half *__restrict__ lhs,
                                            const __half *__restrict__ rhs,
                                            uint8_t *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = static_cast<uint8_t>(a < b);
  }
}

extern "C" int hip_elementwise_less(void *stream, const void *lhs,
                                    const void *rhs, void *output,
                                    int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_less: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_less_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<uint8_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_less: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_less_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<uint8_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_less: i32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_less_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<uint8_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_less: i64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_less_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<uint8_t *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_less: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_less launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// =====================================================================
// Mod (integer / Python-style) and Fmod (FP / C-style).
//
// ORT (common.cuh):
//   _Mod(a, b):  r = a%b; if (sign(r)!=sign(b) && r!=0) r+=b; return r;
//   _Fmod(half a, half b): fmodf((float)a, (float)b)
//
// ONNX `fmod` attribute:
//   fmod=0 -> _Mod (integer types only; sign follows divisor)
//   fmod=1 -> _Fmod (any floating type allowed)
// =====================================================================

template <typename T>
__global__ void elementwise_mod_int_kernel(const T *__restrict__ lhs,
                                           const T *__restrict__ rhs,
                                           T *__restrict__ output,
                                           int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T a = lhs[idx];
    T b = rhs[idx];
    T r = a % b;
    T zero = T(0);
    if ((r > zero && b < zero) || (r < zero && b > zero)) {
      r += b;
    }
    output[idx] = r;
  }
}

__global__ void elementwise_fmod_f16_kernel(const __half *__restrict__ lhs,
                                            const __half *__restrict__ rhs,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half(fmodf(a, b));
  }
}

__global__ void elementwise_fmod_f32_kernel(const float *__restrict__ lhs,
                                            const float *__restrict__ rhs,
                                            float *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = fmodf(lhs[idx], rhs[idx]);
  }
}

extern "C" int hip_elementwise_mod(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype,
                                   int fmod_flag) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  if (fmod_flag) {
    // Floating-point fmod.
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT16:
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_elementwise_mod(fmod): f16, num=%lld\n",
          (long long)num_elements);
      hipLaunchKernelGGL(elementwise_fmod_f16_kernel, dim3(grid_size),
                         dim3(kBlockSize), 0, hip_stream,
                         static_cast<const __half *>(lhs),
                         static_cast<const __half *>(rhs),
                         static_cast<__half *>(output), num_elements);
      break;
    case HIP_DTYPE_FLOAT32:
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_elementwise_mod(fmod): f32, num=%lld\n",
          (long long)num_elements);
      hipLaunchKernelGGL(elementwise_fmod_f32_kernel, dim3(grid_size),
                         dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float *>(lhs),
                         static_cast<const float *>(rhs),
                         static_cast<float *>(output), num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_mod(fmod): unsupported "
              "dtype=%d (FP only)\n",
              hip_dtype);
      return -1;
    }
  } else {
    // Integer Python-style modulo.
    switch (hip_dtype) {
    case HIP_DTYPE_INT32:
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_elementwise_mod(int): i32, num=%lld\n",
          (long long)num_elements);
      hipLaunchKernelGGL(elementwise_mod_int_kernel<int32_t>, dim3(grid_size),
                         dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int32_t *>(lhs),
                         static_cast<const int32_t *>(rhs),
                         static_cast<int32_t *>(output), num_elements);
      break;
    case HIP_DTYPE_INT64:
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_elementwise_mod(int): i64, num=%lld\n",
          (long long)num_elements);
      hipLaunchKernelGGL(elementwise_mod_int_kernel<int64_t>, dim3(grid_size),
                         dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t *>(lhs),
                         static_cast<const int64_t *>(rhs),
                         static_cast<int64_t *>(output), num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_mod(int): unsupported "
              "dtype=%d (INT only)\n",
              hip_dtype);
      return -1;
    }
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_mod launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// =====================================================================
// Or: y = a | b  (bool input -> bool output, 1 byte each).
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.{h,cu}
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(Or, (a | b)),
//                    SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Or, bool))
//
// ORT specializes Or only on `bool`; for bool encoded as uint8 (0 or 1)
// bitwise `|` and logical `||` agree. We keep `|` to match ORT verbatim.
// =====================================================================

__global__ void elementwise_or_kernel(const uint8_t *__restrict__ lhs,
                                      const uint8_t *__restrict__ rhs,
                                      uint8_t *__restrict__ output,
                                      int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(lhs[idx] | rhs[idx]);
  }
}

extern "C" int hip_elementwise_or(void *stream, const void *lhs,
                                  const void *rhs, void *output,
                                  int64_t num_elements) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_elementwise_or: num=%lld\n",
      (long long)num_elements);
  hipLaunchKernelGGL(elementwise_or_kernel, dim3(grid_size),
                     dim3(kBlockSize), 0, hip_stream,
                     static_cast<const uint8_t *>(lhs),
                     static_cast<const uint8_t *>(rhs),
                     static_cast<uint8_t *>(output), num_elements);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_or launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// =====================================================================
// And: y = a & b  (bool input -> bool output, 1 byte each).
//
// Source: onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.{h,cu}
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(And, (a & b)),
//                    SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool))
//
// ORT specializes And only on `bool`; for bool encoded as uint8 (0 or 1)
// bitwise `&` and logical `&&` agree. We keep `&` to match ORT verbatim.
// =====================================================================

__global__ void elementwise_and_kernel(const uint8_t *__restrict__ lhs,
                                       const uint8_t *__restrict__ rhs,
                                       uint8_t *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<uint8_t>(lhs[idx] & rhs[idx]);
  }
}

extern "C" int hip_elementwise_and(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_elementwise_and: num=%lld\n",
      (long long)num_elements);
  hipLaunchKernelGGL(elementwise_and_kernel, dim3(grid_size),
                     dim3(kBlockSize), 0, hip_stream,
                     static_cast<const uint8_t *>(lhs),
                     static_cast<const uint8_t *>(rhs),
                     static_cast<uint8_t *>(output), num_elements);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_and launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// =====================================================================
// Mul / Add / Sub (typed). Dispatched from wrap_elementwise (Mul/Add) and
// wrap_elementwise_sub (Sub) for the int16/int32/int64 path. Same-shape
// only -- broadcasting is materialised upstream via hip_expand before this
// kernel runs.
// =====================================================================

template <typename T>
__global__ void elementwise_mul_kernel(const T *__restrict__ lhs,
                                       const T *__restrict__ rhs,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] * rhs[idx];
  }
}

template <>
__global__ void elementwise_mul_kernel<__half>(const __half *__restrict__ lhs,
                                               const __half *__restrict__ rhs,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half(a * b);
  }
}

template <typename T>
__global__ void elementwise_add_kernel(const T *__restrict__ lhs,
                                       const T *__restrict__ rhs,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] + rhs[idx];
  }
}

template <>
__global__ void elementwise_add_kernel<__half>(const __half *__restrict__ lhs,
                                               const __half *__restrict__ rhs,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half(a + b);
  }
}

extern "C" int hip_elementwise_mul(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
  switch (hip_dtype) {
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(elementwise_mul_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<int64_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(elementwise_mul_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT16:
    hipLaunchKernelGGL(elementwise_mul_kernel<int16_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int16_t *>(lhs),
                       static_cast<const int16_t *>(rhs),
                       static_cast<int16_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(elementwise_mul_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(elementwise_mul_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_elementwise_mul: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_elementwise_add(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
  switch (hip_dtype) {
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(elementwise_add_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<int64_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(elementwise_add_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT16:
    hipLaunchKernelGGL(elementwise_add_kernel<int16_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int16_t *>(lhs),
                       static_cast<const int16_t *>(rhs),
                       static_cast<int16_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(elementwise_add_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(elementwise_add_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_elementwise_add: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

// =====================================================================
// Min / Max (typed). Dispatched from wrap_elementwise for the
// int16/int32/int64 path (tensor_op = MIN / MAX). Same-shape only --
// broadcasting is materialised upstream by wrap_elementwise via
// hip_expand, identical to the Mul/Add integer path above.
//
// Ternary form `(a < b) ? a : b` is used (rather than the HIP built-in
// `min(a, b)`) so the integer specialisations don't accidentally pick up
// a float overload, and so the FP16 path can promote to float without
// special intrinsics.
// =====================================================================

template <typename T>
__global__ void elementwise_min_kernel(const T *__restrict__ lhs,
                                       const T *__restrict__ rhs,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T a = lhs[idx];
    T b = rhs[idx];
    output[idx] = (a < b) ? a : b;
  }
}

template <>
__global__ void elementwise_min_kernel<__half>(const __half *__restrict__ lhs,
                                               const __half *__restrict__ rhs,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half((a < b) ? a : b);
  }
}

template <typename T>
__global__ void elementwise_max_kernel(const T *__restrict__ lhs,
                                       const T *__restrict__ rhs,
                                       T *__restrict__ output,
                                       int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    T a = lhs[idx];
    T b = rhs[idx];
    output[idx] = (a > b) ? a : b;
  }
}

template <>
__global__ void elementwise_max_kernel<__half>(const __half *__restrict__ lhs,
                                               const __half *__restrict__ rhs,
                                               __half *__restrict__ output,
                                               int64_t num_elements) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(lhs[idx]);
    float b = __half2float(rhs[idx]);
    output[idx] = __float2half((a > b) ? a : b);
  }
}

extern "C" int hip_elementwise_min(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
  switch (hip_dtype) {
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(elementwise_min_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<int64_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(elementwise_min_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(elementwise_min_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(elementwise_min_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_elementwise_min: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_elementwise_max(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
  switch (hip_dtype) {
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(elementwise_max_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<int64_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(elementwise_max_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(elementwise_max_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(elementwise_max_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), num_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_elementwise_max: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

// =====================================================================
// Broadcasting binary elementwise (Add / Mul / Min / Max), 4-D.
//
// MIOpen's miopenOpTensor was pathologically slow on gfx1151 for the
// vision-encoder elementwise shapes (measured ~390 ms for a 6.3M-element
// fp16 op), so wrap_elementwise routes the float/half Add/Mul/Min/Max
// path here instead. Operand shapes are 4-D (left-padded with 1 by the
// HipToLLVM lowering); per-axis broadcast is handled by a 0 stride on any
// axis whose extent is 1. Output writes are fully coalesced; broadcast
// reads hit the L2/L1 repeatedly (no expand materialisation needed).
// =====================================================================

namespace {

struct Bcast4DMeta {
  uint32_t out[4];     // output extents, dim 0 outermost .. dim 3 innermost
  uint32_t lstride[4]; // lhs row-major stride per axis (0 = broadcast)
  uint32_t rstride[4]; // rhs row-major stride per axis (0 = broadcast)
};

__device__ __forceinline__ float ew_load(const float *p, uint32_t i) {
  return p[i];
}
__device__ __forceinline__ float ew_load(const __half *p, uint32_t i) {
  return __half2float(p[i]);
}
__device__ __forceinline__ void ew_store(float *p, uint32_t i, float v) {
  p[i] = v;
}
__device__ __forceinline__ void ew_store(__half *p, uint32_t i, float v) {
  p[i] = __float2half(v);
}

__device__ __forceinline__ float ew_to_float(float v) { return v; }
__device__ __forceinline__ float ew_to_float(__half v) {
  return __half2float(v);
}

template <typename T> __device__ __forceinline__ T ew_cast(float v);
template <> __device__ __forceinline__ float ew_cast<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ __half ew_cast<__half>(float v) {
  return __float2half(v);
}

// op: 0=add, 1=mul, 2=min, 3=max
__device__ __forceinline__ float ew_apply(int op, float a, float b) {
  switch (op) {
  case 0:
    return a + b;
  case 1:
    return a * b;
  case 2:
    return a < b ? a : b;
  default:
    return a > b ? a : b;
  }
}

template <typename T>
__global__ void ew_bcast4d_kernel(const T *__restrict__ lhs,
                                   const T *__restrict__ rhs,
                                   T *__restrict__ output, uint32_t num_elements,
                                   int op, Bcast4DMeta meta) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_elements)
    return;
  uint32_t rem = tid;
  uint32_t li = 0, ri = 0;
#pragma unroll
  for (int d = 3; d >= 0; --d) {
    uint32_t coord = rem % meta.out[d];
    rem /= meta.out[d];
    li += coord * meta.lstride[d];
    ri += coord * meta.rstride[d];
  }
  float a = ew_load(lhs, li);
  float b = ew_load(rhs, ri);
  ew_store(output, tid, ew_apply(op, a, b));
}

// =====================================================================
// Vectorised fast path
//
// Two things limit the kernel above on the attention bias/mask add, which is
// where most of a windowed-attention model's elementwise time goes.
//
// `meta.out[d]` is a runtime value, so `rem % meta.out[d]` and
// `rem /= meta.out[d]` are true integer divisions rather than shifts -- four
// of each per element. And one thread per element means a 32-lane wave issues
// 2-byte accesses, moving 64 bytes per memory instruction against a 128-byte
// cache line.
//
// The fast path fixes both. Each thread owns V contiguous elements, chosen so
// V elements are 16 bytes; since the innermost output extent is a multiple of
// V, all V share the outer coordinates and one index computation serves the
// whole group. And the index itself becomes shifts and masks, because the
// host folds the decomposition into the fewest terms that reproduce it.
//
// The folding is what makes this cheap rather than merely cheaper. Axes an
// operand is contiguous over collapse into a single masked term, so a bias of
// shape [1,H,S,S] against [N,H,S,S] costs one AND, and a mask of shape
// [N,1,S,S] costs one AND plus one shift -- not four divisions either way.
// =====================================================================

// At most one term per axis. Each contributes `((t >> shift) & mask) * stride`
// to the operand index; `mask` is all-ones for a term that reaches the
// outermost axis, where no wrap is possible.
struct Bcast4DTerms {
  uint32_t shift[4];
  uint32_t mask[4];
  uint32_t stride[4];
  uint32_t count;
  uint32_t contiguous; // 1 when the operand advances with the output
};

struct Bcast4DFastMeta {
  Bcast4DTerms lhs;
  Bcast4DTerms rhs;
};

__device__ __forceinline__ uint32_t ew_operand_index(uint32_t t,
                                                     const Bcast4DTerms &tm) {
  uint32_t idx = 0;
  for (uint32_t i = 0; i < tm.count; ++i)
    idx += ((t >> tm.shift[i]) & tm.mask[i]) * tm.stride[i];
  return idx;
}

// V * sizeof(T) == 16, so one uint4 covers a thread's whole group.
template <typename T, int V> union EwVec {
  uint4 raw;
  T e[V];
};

template <typename T, int V>
__global__ void ew_bcast4d_fast_kernel(const T *__restrict__ lhs,
                                        const T *__restrict__ rhs,
                                        T *__restrict__ output,
                                        uint32_t num_vectors, int op,
                                        Bcast4DFastMeta meta) {
  const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vectors)
    return;
  const uint32_t t = v * (uint32_t)V;

  EwVec<T, V> a, b, o;
  const uint32_t li = ew_operand_index(t, meta.lhs);
  if (meta.lhs.contiguous)
    a.raw = *reinterpret_cast<const uint4 *>(lhs + li);
  else
#pragma unroll
    for (int i = 0; i < V; ++i)
      a.e[i] = lhs[li];

  const uint32_t ri = ew_operand_index(t, meta.rhs);
  if (meta.rhs.contiguous)
    b.raw = *reinterpret_cast<const uint4 *>(rhs + ri);
  else
#pragma unroll
    for (int i = 0; i < V; ++i)
      b.e[i] = rhs[ri];

#pragma unroll
  for (int i = 0; i < V; ++i)
    o.e[i] = ew_cast<T>(ew_apply(op, ew_to_float(a.e[i]), ew_to_float(b.e[i])));

  *reinterpret_cast<uint4 *>(output + t) = o.raw;
}

} // namespace

namespace {

// log2 of a power of two, or -1 when the value is not one.
int ew_exact_log2(uint64_t v) {
  if (v == 0 || (v & (v - 1)) != 0)
    return -1;
  int s = 0;
  while ((v >> s) != 1)
    ++s;
  return s;
}

// Fold the per-axis index decomposition into the fewest terms that reproduce
// the operand's index, and report whether the result is usable at all.
//
// Axes are walked innermost first. A run of axes the operand is contiguous
// over collapses into one term, because the sum over that run of
// `coord_d * stride_d` is exactly `((t >> inner_shift) & (extent - 1)) *
// run_stride`. An axis of extent 1 contributes nothing and cannot break a
// run; an axis the operand broadcasts over contributes nothing and does.
//
// Fails when a shift or mask the folded form needs is not a power of two, or
// when a stride would misalign a V-element vector. Every failure sends the
// caller back to the general kernel.
bool ew_build_terms(const uint32_t out[4], const uint32_t stride[4], int vec,
                    Bcast4DTerms &tm) {
  tm.count = 0;
  // A contiguous operand is read with one vector load per thread and so has
  // to land on a vector boundary. One that broadcasts over the innermost axis
  // is read as a single scalar and splatted, which constrains nothing.
  tm.contiguous = (stride[3] == 1) ? 1u : 0u;

  uint64_t inner[4]; // product of out[d+1..3]
  inner[3] = 1;
  for (int d = 2; d >= 0; --d)
    inner[d] = inner[d + 1] * out[d + 1];

  for (int d = 3; d >= 0;) {
    if (stride[d] == 0) {
      --d;
      continue;
    }
    const uint64_t run_stride = stride[d];
    uint64_t extent = out[d];
    int top = d;
    while (top > 0) {
      const int nxt = top - 1;
      if (out[nxt] == 1) { // degenerate: no coordinate, no break
        top = nxt;
        continue;
      }
      if ((uint64_t)stride[nxt] != run_stride * extent)
        break;
      top = nxt;
      extent *= out[top];
    }

    const int shift = ew_exact_log2(inner[d]);
    if (shift < 0)
      return false;
    uint32_t mask;
    if (top == 0) {
      mask = 0xFFFFFFFFu; // outermost: t / inner[d] cannot wrap
    } else {
      if (ew_exact_log2(extent) < 0)
        return false;
      mask = (uint32_t)(extent - 1);
    }
    // A vector load needs every term to land on a V-element boundary. The
    // innermost unit-stride run does so because its extent is a multiple of
    // out[3]; any other term must have a stride that is.
    const bool unit_innermost = (run_stride == 1 && shift == 0);
    if (tm.contiguous && !unit_innermost && (run_stride % (uint64_t)vec) != 0)
      return false;
    if (tm.count == 4)
      return false;
    tm.shift[tm.count] = (uint32_t)shift;
    tm.mask[tm.count] = mask;
    tm.stride[tm.count] = (uint32_t)run_stride;
    ++tm.count;
    d = top - 1;
  }
  return true;
}

// Both operands must fold, the innermost extent must fill whole vectors, and
// the buffers must be 16-byte aligned.
bool ew_bcast4d_fast_meta(const uint32_t out[4], const Bcast4DMeta &meta,
                          int vec, const void *lhs, const void *rhs,
                          const void *output, uint32_t n,
                          Bcast4DFastMeta &fast) {
  if (out[3] % (uint32_t)vec != 0 || n % (uint32_t)vec != 0)
    return false;
  const uintptr_t bytes = 16;
  if ((reinterpret_cast<uintptr_t>(lhs) % bytes) != 0 ||
      (reinterpret_cast<uintptr_t>(rhs) % bytes) != 0 ||
      (reinterpret_cast<uintptr_t>(output) % bytes) != 0)
    return false;
  return ew_build_terms(out, meta.lstride, vec, fast.lhs) &&
         ew_build_terms(out, meta.rstride, vec, fast.rhs);
}

} // namespace

// Returns 0 on success, -2 when the output volume exceeds the 32-bit index
// range (caller falls back to the MIOpen path).
extern "C" int hip_elementwise_binary_bcast(void *stream, const void *lhs,
                                            const void *rhs, void *output,
                                            const int64_t *lhs_shape4,
                                            const int64_t *rhs_shape4,
                                            const int64_t *out_shape4, int op,
                                            int hip_dtype) {
  int64_t out_vol = out_shape4[0] * out_shape4[1] * out_shape4[2] * out_shape4[3];
  if (out_vol <= 0)
    return 0;
  if (out_vol > static_cast<int64_t>(0xFFFFFFFFu))
    return -2;

  Bcast4DMeta meta{};
  // Row-major contiguous strides derived from each operand's own 4-D shape;
  // zeroed on any axis whose extent is 1 so that axis broadcasts.
  int64_t lhs_contig[4], rhs_contig[4];
  lhs_contig[3] = 1;
  rhs_contig[3] = 1;
  for (int d = 2; d >= 0; --d) {
    lhs_contig[d] = lhs_contig[d + 1] * lhs_shape4[d + 1];
    rhs_contig[d] = rhs_contig[d + 1] * rhs_shape4[d + 1];
  }
  for (int d = 0; d < 4; ++d) {
    meta.out[d] = static_cast<uint32_t>(out_shape4[d]);
    meta.lstride[d] =
        (lhs_shape4[d] == 1) ? 0u : static_cast<uint32_t>(lhs_contig[d]);
    meta.rstride[d] =
        (rhs_shape4[d] == 1) ? 0u : static_cast<uint32_t>(rhs_contig[d]);
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  const uint32_t n = static_cast<uint32_t>(out_vol);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((out_vol + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();

  // 16 bytes per thread: 8 halves or 4 floats.
  Bcast4DFastMeta fast{};
  const int vec = (hip_dtype == HIP_DTYPE_FLOAT16)   ? 8
                  : (hip_dtype == HIP_DTYPE_FLOAT32) ? 4
                                                     : 0;
  if (vec && ew_bcast4d_fast_meta(meta.out, meta, vec, lhs, rhs, output, n,
                                  fast)) {
    const uint32_t num_vectors = n / (uint32_t)vec;
    const int fast_grid =
        static_cast<int>((num_vectors + kBlockSize - 1) / kBlockSize);
    if (hip_dtype == HIP_DTYPE_FLOAT16)
      hipLaunchKernelGGL(
          (ew_bcast4d_fast_kernel<__half, 8>), dim3(fast_grid),
          dim3(kBlockSize), 0, hip_stream, static_cast<const __half *>(lhs),
          static_cast<const __half *>(rhs), static_cast<__half *>(output),
          num_vectors, op, fast);
    else
      hipLaunchKernelGGL(
          (ew_bcast4d_fast_kernel<float, 4>), dim3(fast_grid), dim3(kBlockSize),
          0, hip_stream, static_cast<const float *>(lhs),
          static_cast<const float *>(rhs), static_cast<float *>(output),
          num_vectors, op, fast);
    hipError_t ferr = hipGetLastError();
    if (ferr != hipSuccess)
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_binary_bcast fast path launch "
              "failed: %s\n",
              hipGetErrorString(ferr));
    return static_cast<int>(ferr);
  }

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(ew_bcast4d_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), n, op, meta);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(ew_bcast4d_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), n, op, meta);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_binary_bcast: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_binary_bcast launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_elementwise_div(void *stream, const void *lhs,
                                   const void *rhs, void *output,
                                   int64_t num_elements, int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_div: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_div_kernel<__half>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(lhs),
                       static_cast<const __half *>(rhs),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_div: f32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_div_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(lhs),
                       static_cast<const float *>(rhs),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_div: i32, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_div_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(lhs),
                       static_cast<const int32_t *>(rhs),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_div: i64, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_div_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(lhs),
                       static_cast<const int64_t *>(rhs),
                       static_cast<int64_t *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_div: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_div launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
