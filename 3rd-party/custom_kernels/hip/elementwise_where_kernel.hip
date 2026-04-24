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

// Fixed-size shape/stride buffers passed as kernel arguments (by value).
// Capacity is HIP_WHERE_MAX_RANK; entries past `rank` are unused.
struct WhereDimsND {
  int64_t v[HIP_WHERE_MAX_RANK];
};

// N-D broadcast kernel.
// Decomposes the linear output index into coordinates along `out_shape.v[0..rank)`
// using row-major order, then reconstructs a linear index into each operand
// using its precomputed broadcast stride (stride == 0 along axes where the
// operand's padded dim is 1, i.e. broadcast).
template <typename T>
__global__ void where_nd_kernel(const bool *__restrict__ condition,
                                const T *__restrict__ x,
                                const T *__restrict__ y,
                                T *__restrict__ output,
                                int64_t total_elements, int rank,
                                WhereDimsND out_shape, WhereDimsND cond_stride,
                                WhereDimsND x_stride, WhereDimsND y_stride) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  int64_t cond_idx = 0, x_idx = 0, y_idx = 0;
  int64_t tmp = idx;
  for (int k = rank - 1; k >= 0; --k) {
    int64_t dim = out_shape.v[k];
    int64_t coord = tmp % dim;
    tmp /= dim;
    cond_idx += coord * cond_stride.v[k];
    x_idx += coord * x_stride.v[k];
    y_idx += coord * y_stride.v[k];
  }

  output[idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
}

// Left-pad `src_shape` (length `src_rank`) to `out_rank` with 1s into
// `padded_shape`, then compute row-major strides over the padded shape with
// stride 0 along axes where padded dim is 1 (NumPy-style broadcast).
static bool make_broadcast_layout(const int64_t *src_shape, int src_rank,
                                  int out_rank, const int64_t *out_shape,
                                  WhereDimsND *padded_shape,
                                  WhereDimsND *stride, const char *name) {
  int pad = out_rank - src_rank;
  for (int i = 0; i < pad; ++i)
    padded_shape->v[i] = 1;
  for (int i = 0; i < src_rank; ++i)
    padded_shape->v[pad + i] = src_shape[i];

  int64_t running = 1;
  for (int k = out_rank - 1; k >= 0; --k) {
    int64_t dim = padded_shape->v[k];
    if (dim == 1) {
      stride->v[k] = 0;
    } else if (dim == out_shape[k]) {
      stride->v[k] = running;
    } else {
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_where: %s dim %d = %lld is "
              "not broadcast-compatible with out dim %lld\n",
              name, k, (long long)dim, (long long)out_shape[k]);
      return false;
    }
    running *= dim;
  }
  return true;
}

template <typename T>
static int launch_where(hipStream_t stream, const void *condition,
                        const void *x, const void *y, void *output,
                        int64_t total, int out_rank,
                        const WhereDimsND &out_shape,
                        const WhereDimsND &cond_stride,
                        const WhereDimsND &x_stride,
                        const WhereDimsND &y_stride) {
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(where_nd_kernel<T>, dim3(grid_size), dim3(kBlockSize), 0,
                     stream, static_cast<const bool *>(condition),
                     static_cast<const T *>(x), static_cast<const T *>(y),
                     static_cast<T *>(output), total, out_rank, out_shape,
                     cond_stride, x_stride, y_stride);
  return 0;
}

extern "C" int hip_elementwise_where(
    void *stream,
    const void *condition,
    const void *x,
    const void *y,
    void *output,
    const int64_t *cond_shape, int64_t cond_rank,
    const int64_t *x_shape,    int64_t x_rank,
    const int64_t *y_shape,    int64_t y_rank,
    const int64_t *out_shape,  int64_t out_rank,
    int hip_dtype) {
  if (out_rank < 0 || out_rank > HIP_WHERE_MAX_RANK || cond_rank < 0 ||
      cond_rank > out_rank || x_rank < 0 || x_rank > out_rank || y_rank < 0 ||
      y_rank > out_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where: invalid ranks "
            "(max=%d): cond=%lld, x=%lld, y=%lld, out=%lld\n",
            HIP_WHERE_MAX_RANK, (long long)cond_rank, (long long)x_rank,
            (long long)y_rank, (long long)out_rank);
    return -1;
  }

  int64_t total = 1;
  for (int i = 0; i < out_rank; ++i)
    total *= out_shape[i];
  if (total <= 0)
    return 0;

  WhereDimsND out_shape_nd = {};
  WhereDimsND cond_shape_nd = {}, x_shape_nd = {}, y_shape_nd = {};
  WhereDimsND cond_stride = {}, x_stride = {}, y_stride = {};
  for (int i = 0; i < out_rank; ++i)
    out_shape_nd.v[i] = out_shape[i];

  if (!make_broadcast_layout(cond_shape, (int)cond_rank, (int)out_rank,
                             out_shape, &cond_shape_nd, &cond_stride,
                             "condition"))
    return -1;
  if (!make_broadcast_layout(x_shape, (int)x_rank, (int)out_rank, out_shape,
                             &x_shape_nd, &x_stride, "x"))
    return -1;
  if (!make_broadcast_layout(y_shape, (int)y_rank, (int)out_rank, out_shape,
                             &y_shape_nd, &y_stride, "y"))
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_elementwise_where: dtype=%d, out_rank=%d, "
      "total=%lld\n",
      hip_dtype, (int)out_rank, (long long)total);

  int rc = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    rc = launch_where<float>(hip_stream, condition, x, y, output, total,
                             (int)out_rank, out_shape_nd, cond_stride, x_stride,
                             y_stride);
    break;
  case HIP_DTYPE_FLOAT16:
    rc = launch_where<__half>(hip_stream, condition, x, y, output, total,
                              (int)out_rank, out_shape_nd, cond_stride,
                              x_stride, y_stride);
    break;
  case HIP_DTYPE_BFLOAT16:
    rc = launch_where<__hip_bfloat16>(hip_stream, condition, x, y, output,
                                      total, (int)out_rank, out_shape_nd,
                                      cond_stride, x_stride, y_stride);
    break;
  case HIP_DTYPE_INT32:
    rc = launch_where<int32_t>(hip_stream, condition, x, y, output, total,
                               (int)out_rank, out_shape_nd, cond_stride,
                               x_stride, y_stride);
    break;
  case HIP_DTYPE_INT64:
    rc = launch_where<int64_t>(hip_stream, condition, x, y, output, total,
                               (int)out_rank, out_shape_nd, cond_stride,
                               x_stride, y_stride);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_where launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
