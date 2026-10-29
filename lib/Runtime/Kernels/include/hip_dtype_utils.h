/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Shared dtype helpers for the custom HIP kernels (include from .hip only).
 *
 *   DTypeIO<T>             floating types: load() to ComputeType, store()
 *                          back. ComputeType is float for f32/f16/bf16 and
 *                          double for f64. Undefined for integer types.
 *   CastIO<T>              Cast only: any supported type <-> float.
 *   dispatch_float_dtype() runtime dtype -> f32/f16/bf16/f64.
 *   dispatch_dtype()       runtime dtype -> floating or integer type.
 *
 * Dispatchers call f(TypeTag<T>{}) and return its result, or -1 for an
 * unsupported dtype. f16/bf16 conversions use the HIP intrinsics, which
 * round to nearest even (same as PyTorch and ONNX).
 *
 * Namespaced so it cannot clash with per-kernel helpers (e.g. DTypeIO in
 * gather_block_quantized_kernel.hip).
 */

#ifndef HIP_DTYPE_UTILS_H
#define HIP_DTYPE_UTILS_H

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace hip_dtype {

template <typename T>
struct TypeTag {
  using type = T;
};

// Floating types only.
template <typename T>
struct DTypeIO;

template <>
struct DTypeIO<float> {
  using ComputeType = float;
  __device__ static inline float load(float x) { return x; }
  __device__ static inline float store(float x) { return x; }
};

template <>
struct DTypeIO<__half> {
  using ComputeType = float;
  __device__ static inline float load(__half x) { return __half2float(x); }
  __device__ static inline __half store(float x) { return __float2half(x); }
};

template <>
struct DTypeIO<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static inline float load(__hip_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static inline __hip_bfloat16 store(float x) {
    return __float2bfloat16(x);
  }
};

template <>
struct DTypeIO<double> {
  using ComputeType = double;
  __device__ static inline double load(double x) { return x; }
  __device__ static inline double store(double x) { return x; }
};

// Cast only: integers and f64 via static_cast, f16/bf16 reuse DTypeIO.
template <typename T>
struct CastIO {
  __device__ static inline float load(T x) { return static_cast<float>(x); }
  __device__ static inline T store(float x) { return static_cast<T>(x); }
};

template <>
struct CastIO<__half> : DTypeIO<__half> {};

template <>
struct CastIO<__hip_bfloat16> : DTypeIO<__hip_bfloat16> {};

template <typename F>
inline int dispatch_float_dtype(int dtype, F&& f) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT32:  return f(TypeTag<float>{});
  case HIP_DTYPE_FLOAT16:  return f(TypeTag<__half>{});
  case HIP_DTYPE_BFLOAT16: return f(TypeTag<__hip_bfloat16>{});
  case HIP_DTYPE_FLOAT64:  return f(TypeTag<double>{});
  default:                 return -1;
  }
}

template <typename F>
inline int dispatch_dtype(int dtype, F&& f) {
  switch (dtype) {
  case HIP_DTYPE_INT64: return f(TypeTag<int64_t>{});
  case HIP_DTYPE_INT32: return f(TypeTag<int32_t>{});
  case HIP_DTYPE_INT16: return f(TypeTag<int16_t>{});
  case HIP_DTYPE_INT8:  return f(TypeTag<int8_t>{});
  case HIP_DTYPE_UINT8: return f(TypeTag<uint8_t>{});
  default:              return dispatch_float_dtype(dtype, f);
  }
}

} // namespace hip_dtype

#endif // HIP_DTYPE_UTILS_H
