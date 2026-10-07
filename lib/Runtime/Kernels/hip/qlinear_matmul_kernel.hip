/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX QLinearMatMul, rank 2, one output element per thread.
//
//   dot = matmul(a - a_zp, b - b_zp)
//   y = saturate(round(dot * a_scale * b_scale / y_scale) + y_zp)
//
// a is [M, K], b is [K, N], y is [M, N], all row-major. Scales and zero
// points are per-tensor. The dot product accumulates in int64 so the only
// rounding is the final requantization, which uses rintf (ties to even),
// matching QuantizeLinear.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

template <typename T> __device__ float qlinear_mm_lowest();
template <typename T> __device__ float qlinear_mm_highest();

template <> __device__ float qlinear_mm_lowest<uint8_t>() { return 0.0f; }
template <> __device__ float qlinear_mm_highest<uint8_t>() { return 255.0f; }
template <> __device__ float qlinear_mm_lowest<int8_t>() { return -128.0f; }
template <> __device__ float qlinear_mm_highest<int8_t>() { return 127.0f; }

template <typename AT, typename BT, typename YT>
__global__ void qlinear_matmul_2d(const AT *a, const float *a_scale,
                                  const AT *a_zp, const BT *b,
                                  const float *b_scale, const BT *b_zp,
                                  const float *y_scale, const YT *y_zp, YT *y,
                                  int64_t m, int64_t k, int64_t n) {
  const int64_t total = m * n;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float as = a_scale[0];
  const float bs = b_scale[0];
  const float ys = y_scale[0];
  const int a_zp_v = static_cast<int>(a_zp[0]);
  const int b_zp_v = static_cast<int>(b_zp[0]);
  const int y_zp_v = static_cast<int>(y_zp[0]);
  const float scale = as * bs / ys;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t col = idx % n;
    const int64_t row = idx / n;
    int64_t dot = 0;
    const AT *a_row = a + row * k;
    for (int64_t ki = 0; ki < k; ++ki) {
      const int av = static_cast<int>(a_row[ki]) - a_zp_v;
      const int bv = static_cast<int>(b[ki * n + col]) - b_zp_v;
      dot += static_cast<int64_t>(av) * static_cast<int64_t>(bv);
    }
    float q = rintf(static_cast<float>(dot) * scale) +
              static_cast<float>(y_zp_v);
    q = fminf(qlinear_mm_highest<YT>(), fmaxf(qlinear_mm_lowest<YT>(), q));
    y[idx] = static_cast<YT>(static_cast<int>(q));
  }
}

template <typename AT, typename BT, typename YT>
int launch_qlinear_matmul(hipStream_t stream, const void *a,
                          const void *a_scale, const void *a_zp, const void *b,
                          const void *b_scale, const void *b_zp,
                          const void *y_scale, const void *y_zp, void *y,
                          int64_t m, int64_t k, int64_t n) {
  const int64_t total = m * n;
  if (total <= 0)
    return 0;
  constexpr int kThreads = 256;
  int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 65535)
    blocks = 65535;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(qlinear_matmul_2d<AT, BT, YT>),
                     dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0,
                     stream, static_cast<const AT *>(a),
                     static_cast<const float *>(a_scale),
                     static_cast<const AT *>(a_zp), static_cast<const BT *>(b),
                     static_cast<const float *>(b_scale),
                     static_cast<const BT *>(b_zp),
                     static_cast<const float *>(y_scale),
                     static_cast<const YT *>(y_zp), static_cast<YT *>(y), m, k,
                     n);
  return 0;
}

int hip_dtype_pair(int dtype, bool *is_signed) {
  if (dtype == HIP_DTYPE_UINT8) {
    *is_signed = false;
    return 0;
  }
  if (dtype == HIP_DTYPE_INT8) {
    *is_signed = true;
    return 0;
  }
  return -1;
}

} // namespace

int hip_qlinear_matmul(void *stream, const void *a, const void *a_scale,
                       const void *a_zero_point, const void *b,
                       const void *b_scale, const void *b_zero_point,
                       const void *y_scale, const void *y_zero_point, void *y,
                       int64_t m, int64_t k, int64_t n, int a_dtype, int b_dtype,
                       int y_dtype) {
  bool a_signed = false;
  bool b_signed = false;
  bool y_signed = false;
  if (hip_dtype_pair(a_dtype, &a_signed) || hip_dtype_pair(b_dtype, &b_signed) ||
      hip_dtype_pair(y_dtype, &y_signed)) {
    fprintf(stderr,
            "[custom_kernels] hip_qlinear_matmul: unsupported dtypes "
            "a=%d b=%d y=%d\n",
            a_dtype, b_dtype, y_dtype);
    return -1;
  }

  using LaunchFn =
      int (*)(hipStream_t, const void *, const void *, const void *,
              const void *, const void *, const void *, const void *,
              const void *, void *, int64_t, int64_t, int64_t);
  LaunchFn launch = nullptr;
  if (!a_signed && b_signed && !y_signed)
    launch = launch_qlinear_matmul<uint8_t, int8_t, uint8_t>;
  else if (!a_signed && !b_signed && !y_signed)
    launch = launch_qlinear_matmul<uint8_t, uint8_t, uint8_t>;
  else if (a_signed && b_signed && !y_signed)
    launch = launch_qlinear_matmul<int8_t, int8_t, uint8_t>;
  else if (!a_signed && b_signed && y_signed)
    launch = launch_qlinear_matmul<uint8_t, int8_t, int8_t>;
  else if (a_signed && !b_signed && !y_signed)
    launch = launch_qlinear_matmul<int8_t, uint8_t, uint8_t>;
  else if (a_signed && b_signed && y_signed)
    launch = launch_qlinear_matmul<int8_t, int8_t, int8_t>;
  else if (!a_signed && !b_signed && y_signed)
    launch = launch_qlinear_matmul<uint8_t, uint8_t, int8_t>;
  else
    launch = launch_qlinear_matmul<int8_t, uint8_t, int8_t>;

  (void)hipGetLastError();
  int rc = launch(static_cast<hipStream_t>(stream), a, a_scale, a_zero_point, b,
                  b_scale, b_zero_point, y_scale, y_zero_point, y, m, k, n);
  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qlinear_matmul launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
