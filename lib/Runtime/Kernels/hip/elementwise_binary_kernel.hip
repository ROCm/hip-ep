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
#include <hip/hip_bf16.h>
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
// Mul / Add / Sub (typed). Dispatched from wrap_miopenOpTensor as the
// integer-dtype fallback, because MIOpen's miopenOpTensor doesn't support
// integer element types. Same-shape only -- broadcasting is materialised
// upstream (see wrap_miopenOpTensor's 2-D broadcast fallback for the
// general case; this kernel is only reached when shapes already match).
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
// Min / Max (typed). Dispatched from wrap_miopenOpTensor as the
// integer-dtype fallback for tensor_op = MIN / MAX, because MIOpen's
// miopenOpTensor only supports floating-point element types. Same-shape
// only -- broadcasting is materialised upstream by wrap_miopenOpTensor
// via hip_expand, identical to the Mul/Add integer path above.
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
// MIOpen's miopenOpTensor is pathologically slow on gfx1151 for the
// vision-encoder elementwise shapes (measured ~390 ms for a 6.3M-element
// fp16 op), so wrap_miopenOpTensor routes the float/half Add/Mul/Min/Max
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
__device__ __forceinline__ float ew_load(const __hip_bfloat16 *p, uint32_t i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void ew_store(float *p, uint32_t i, float v) {
  p[i] = v;
}
__device__ __forceinline__ void ew_store(__half *p, uint32_t i, float v) {
  p[i] = __float2half(v);
}
__device__ __forceinline__ void ew_store(__hip_bfloat16 *p, uint32_t i,
                                         float v) {
  p[i] = __float2bfloat16(v);
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
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(ew_bcast4d_kernel<__hip_bfloat16>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(lhs),
                       static_cast<const __hip_bfloat16 *>(rhs),
                       static_cast<__hip_bfloat16 *>(output), n, op, meta);
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
