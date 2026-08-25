/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * MIOpen miopenOpTensor (MIN) de-library path for hip-ep.
 *
 * Ports the device-side logic from MIOpen's MIOpenTensorKernelsHip.cpp:
 *   - miopenMin / Op4dTensorLite (packed equal-shape tensors)
 *   - Op4dTensorGeneric (single-side broadcast via bitmap)
 *
 * Host entry: hip_miopen_op_tensor_min implements
 *   C = min(alpha0 * A, alpha1 * B) + beta * C  (hip-ep uses 1, 1, 0).
 *
 * The runtime wrapper remains wrap_miopenOpTensor to mark the technical path.
 *
 * Source reference:
 *   rocm-libraries/projects/miopen/src/kernels/MIOpenTensorKernelsHip.cpp
 *   rocm-libraries/projects/miopen/src/solver/tensorOp/tensor_op_helpers.hpp
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace {

using dim_t = uint32_t;

constexpr dim_t kMaxNumWg = 4096;
constexpr dim_t kBlockThreads = 256;

// ---------------------------------------------------------------------------
// Device helpers (ported from MIOpenTensorKernelsHip.cpp)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T miopen_min(T a, T b) {
  return (a < b) ? a : b;
}

__device__ __forceinline__ float load_as_float(float v) { return v; }
__device__ __forceinline__ float load_as_float(__half v) {
  return __half2float(v);
}
__device__ __forceinline__ float load_as_float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(v);
  } else {
    return __float2bfloat16(v);
  }
}

// Op4dTensorLite (USE_4D_TENSOR_LITE, beta == 0, RD_BLCK == 1 specialization).
// MIOpen vectorizes with RD_BLCK>1 for larger tensors; RD_BLCK=1 preserves
// semantics and supports int64 element counts beyond the 32-bit bcast kernel.
template <typename T>
__global__ void hip_miopen_op4d_tensor_lite_min(const T *__restrict__ a,
                                                const T *__restrict__ b,
                                                T *__restrict__ c, float alpha0,
                                                float alpha1, int64_t total_work) {
  const int64_t gid0 = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  const int64_t global_size =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t gid = gid0; gid < total_work; gid += global_size) {
    const float av = load_as_float(a[gid]) * alpha0;
    const float bv = load_as_float(b[gid]) * alpha1;
    c[gid] = from_float<T>(miopen_min(av, bv));
  }
}

// Op4dTensorGeneric (USE_4D_TENSOR_GENERIC, beta == 0, MIN only).
// NCHW layout; `bitmap` marks which B-tensor axes are broadcast (1-extent).
template <typename T>
__global__ void hip_miopen_op4d_tensor_generic_min(
    const T *__restrict__ a, dim_t a_nstride, dim_t a_cstride, dim_t a_hstride,
    const T *__restrict__ b, dim_t b_c, dim_t b_h, dim_t b_w, dim_t b_nstride,
    dim_t b_cstride, dim_t b_hstride, T *__restrict__ c, dim_t c_c, dim_t c_h,
    dim_t c_w, dim_t c_nstride, dim_t c_cstride, dim_t c_hstride, float alpha0,
    float alpha1, unsigned int bitmap, dim_t work_per_wg, dim_t num_wg) {
  dim_t gid = blockIdx.x;

  for (; gid < num_wg; gid += kMaxNumWg) {
    dim_t lid = threadIdx.x;

    const dim_t o_h_div = (bitmap & (1u << 0)) ? 1u : c_w;
    const dim_t o_c_div = o_h_div * ((bitmap & (1u << 1)) ? 1u : c_h);
    const dim_t o_n_div = o_c_div * ((bitmap & (1u << 2)) ? 1u : c_c);

    const dim_t o_w_gid_off = gid % b_w;
    const dim_t o_h_gid_off = (gid / b_w) % b_h;
    const dim_t o_c_gid_off = (gid / b_w / b_h) % b_c;
    const dim_t o_n_gid_off = (gid / b_w / b_h) / b_c;

    const dim_t bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
    const float operand = load_as_float(b[bindex]) * alpha1;

    while (lid < work_per_wg) {
      const dim_t o_w =
          (bitmap & (1u << 0)) ? o_w_gid_off : lid % c_w;
      const dim_t o_h =
          (bitmap & (1u << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
      const dim_t o_c =
          (bitmap & (1u << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
      const dim_t o_n =
          (bitmap & (1u << 3)) ? o_n_gid_off : lid / o_n_div;

      const dim_t aindex =
          o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
      const dim_t cindex =
          o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
      const float aval = load_as_float(a[aindex]) * alpha0;
      c[cindex] = from_float<T>(miopen_min(aval, operand));

      lid += blockDim.x;
    }
  }
}

struct PackedStrides {
  dim_t n;
  dim_t c;
  dim_t h;
};

static void packed_nchw_strides(int64_t n, int64_t c, int64_t h, int64_t w,
                                PackedStrides &out) {
  out.h = static_cast<dim_t>(w);
  out.c = static_cast<dim_t>(h * w);
  out.n = static_cast<dim_t>(c * h * w);
}

struct BcastParams {
  int num_wg_orig;
  dim_t num_wg;
  dim_t work_per_wg;
  unsigned int bitmap;
};

// Port of GetBitmapAndWgInfo + Op4dTensorGeneric quick fixes from MIOpen.
static BcastParams compute_bcast_params(int64_t b_n, int64_t b_c, int64_t b_h,
                                        int64_t b_w, int64_t c_n, int64_t c_c,
                                        int64_t c_h, int64_t c_w) {
  const int64_t blens[4] = {b_n, b_c, b_h, b_w};
  const int64_t clens[4] = {c_n, c_c, c_h, c_w};

  int d = 4;
  for (int i = 3; i >= 0; --i) {
    if (blens[i] != 1) {
      d = i;
      break;
    }
  }

  int num_wg = 1;
  if (d < 4) {
    num_wg = static_cast<int>(blens[d] == 0 ? 1 : blens[d]);
  }

  int64_t work_per_wg64 = 1;
  for (int i = d; i < 4; ++i) {
    work_per_wg64 *= clens[i];
  }

  unsigned int bitmap = 0;
  if (d < 4) {
    bitmap |= (1u << (4 - d));
  }

  for (int i = d - 1; i >= 0; --i) {
    if (blens[i] != 1) {
      bitmap |= (1u << (3 - i));
      num_wg *= static_cast<int>(blens[i]);
    } else {
      work_per_wg64 *= clens[i];
    }
  }

  const int64_t b_vol = b_n * b_c * b_h * b_w;
  if (b_vol == 1) {
    bitmap = 4;
  }

  BcastParams p{};
  p.num_wg_orig = num_wg;
  p.num_wg = static_cast<dim_t>(num_wg > static_cast<int>(kMaxNumWg)
                                    ? kMaxNumWg
                                    : num_wg);
  p.work_per_wg = static_cast<dim_t>(work_per_wg64);
  p.bitmap = bitmap;
  return p;
}

static bool shapes_equal4(int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                            int64_t b0, int64_t b1, int64_t b2, int64_t b3) {
  return a0 == b0 && a1 == b1 && a2 == b2 && a3 == b3;
}

template <typename T>
static int launch_lite(void *stream, const void *a, const void *b, void *c,
                       float alpha0, float alpha1, int64_t total_work) {
  if (total_work <= 0) {
    return 0;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlock = 256;
  const int64_t grid64 = (total_work + kBlock - 1) / kBlock;
  const int grid =
      static_cast<int>(grid64 > INT32_MAX ? INT32_MAX : grid64);

  (void)hipGetLastError();
  hipLaunchKernelGGL(hip_miopen_op4d_tensor_lite_min<T>, dim3(grid),
                     dim3(kBlock), 0, hip_stream,
                     static_cast<const T *>(a), static_cast<const T *>(b),
                     static_cast<T *>(c), alpha0, alpha1, total_work);

  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_miopen_op4d_tensor_lite_min launch failed: "
            "%s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

template <typename T>
static int launch_generic(void *stream, const void *a, const void *b, void *c,
                          int64_t a_n, int64_t a_c, int64_t a_h, int64_t a_w,
                          int64_t b_n, int64_t b_c, int64_t b_h, int64_t b_w,
                          int64_t c_n, int64_t c_c, int64_t c_h, int64_t c_w,
                          float alpha0, float alpha1, const BcastParams &bp) {
  PackedStrides a_strides, b_strides, c_strides;
  packed_nchw_strides(a_n, a_c, a_h, a_w, a_strides);
  packed_nchw_strides(b_n, b_c, b_h, b_w, b_strides);
  packed_nchw_strides(c_n, c_c, c_h, c_w, c_strides);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  (void)hipGetLastError();
  hipLaunchKernelGGL(
      hip_miopen_op4d_tensor_generic_min<T>, dim3(bp.num_wg),
      dim3(kBlockThreads), 0, hip_stream, static_cast<const T *>(a),
      a_strides.n, a_strides.c, a_strides.h, static_cast<const T *>(b),
      static_cast<dim_t>(b_c), static_cast<dim_t>(b_h),
      static_cast<dim_t>(b_w), b_strides.n, b_strides.c, b_strides.h,
      static_cast<T *>(c), static_cast<dim_t>(c_c), static_cast<dim_t>(c_h),
      static_cast<dim_t>(c_w), c_strides.n, c_strides.c, c_strides.h, alpha0,
      alpha1, bp.bitmap, bp.work_per_wg,
      static_cast<dim_t>(bp.num_wg_orig));

  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_miopen_op4d_tensor_generic_min launch "
            "failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

template <typename T>
static int dispatch_min(void *stream, const void *a, const void *b, void *c,
                        int64_t a_n, int64_t a_c, int64_t a_h, int64_t a_w,
                        int64_t b_n, int64_t b_c, int64_t b_h, int64_t b_w,
                        int64_t c_n, int64_t c_c, int64_t c_h, int64_t c_w,
                        float alpha0, float alpha1) {
  const int64_t c_vol = c_n * c_c * c_h * c_w;
  if (c_vol <= 0) {
    return 0;
  }

  const bool packed_equal =
      shapes_equal4(a_n, a_c, a_h, a_w, b_n, b_c, b_h, b_w) &&
      shapes_equal4(a_n, a_c, a_h, a_w, c_n, c_c, c_h, c_w);

  if (packed_equal) {
    return launch_lite<T>(stream, a, b, c, alpha0, alpha1, c_vol);
  }

  const BcastParams bp =
      compute_bcast_params(b_n, b_c, b_h, b_w, c_n, c_c, c_h, c_w);
  return launch_generic<T>(stream, a, b, c, a_n, a_c, a_h, a_w, b_n, b_c, b_h,
                           b_w, c_n, c_c, c_h, c_w, alpha0, alpha1, bp);
}

} // namespace

extern "C" int hip_miopen_op_tensor_min(
    void *stream, const void *a, const void *b, void *c, int64_t a_n,
    int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
    int64_t b_h, int64_t b_w, int64_t c_n, int64_t c_c, int64_t c_h,
    int64_t c_w, int hip_dtype, float alpha0, float alpha1, float beta) {
  (void)beta; // hip-ep Min uses beta=0; kernel paths assume beta == 0.

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return dispatch_min<float>(stream, a, b, c, a_n, a_c, a_h, a_w, b_n, b_c,
                               b_h, b_w, c_n, c_c, c_h, c_w, alpha0, alpha1);
  case HIP_DTYPE_FLOAT16:
    return dispatch_min<__half>(stream, a, b, c, a_n, a_c, a_h, a_w, b_n, b_c,
                               b_h, b_w, c_n, c_c, c_h, c_w, alpha0, alpha1);
  case HIP_DTYPE_BFLOAT16:
    return dispatch_min<__hip_bfloat16>(
        stream, a, b, c, a_n, a_c, a_h, a_w, b_n, b_c, b_h, b_w, c_n, c_c,
        c_h, c_w, alpha0, alpha1);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_miopen_op_tensor_min: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
