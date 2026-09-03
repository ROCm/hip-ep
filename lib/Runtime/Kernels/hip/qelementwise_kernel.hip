/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#define HIP_QELEMENTWISE_MAX_RANK 8
#define HIP_QELEMENTWISE_ADD 0

struct QElemDimsND {
  int64_t v[HIP_QELEMENTWISE_MAX_RANK];
};

template <typename T>
__device__ inline T q_saturate(float v) {
  float lo = static_cast<float>(std::numeric_limits<T>::lowest());
  float hi = static_cast<float>(std::numeric_limits<T>::max());
  return static_cast<T>(fminf(hi, fmaxf(lo, v)));
}

template <typename T>
__global__ void qelementwise_nd_kernel(
    const T *__restrict__ lhs, const T *__restrict__ rhs, T *__restrict__ output,
    int64_t total_elements, int rank, int kind, float M_a, float M_b,
    int32_t lhs_zp, int32_t rhs_zp, int32_t output_zp, QElemDimsND out_shape,
    QElemDimsND lhs_stride, QElemDimsND rhs_stride) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  int64_t lhs_idx = 0, rhs_idx = 0;
  int64_t tmp = idx;
  for (int k = rank - 1; k >= 0; --k) {
    int64_t dim = out_shape.v[k];
    int64_t coord = tmp % dim;
    tmp /= dim;
    lhs_idx += coord * lhs_stride.v[k];
    rhs_idx += coord * rhs_stride.v[k];
  }

  float da = M_a * (static_cast<float>(lhs[lhs_idx]) -
                    static_cast<float>(lhs_zp));
  float db = M_b * (static_cast<float>(rhs[rhs_idx]) -
                    static_cast<float>(rhs_zp));
  float acc = 0.f;
  if (kind == HIP_QELEMENTWISE_ADD)
    acc = da + db;
  output[idx] = q_saturate<T>(rintf(acc) + static_cast<float>(output_zp));
}

static bool make_broadcast_layout(const int64_t *src_shape, int src_rank,
                                  int out_rank, const int64_t *out_shape,
                                  QElemDimsND *padded_shape, QElemDimsND *stride,
                                  const char *name) {
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
              "[custom_kernels] hip_qelementwise: %s dim %d = %lld is "
              "not broadcast-compatible with out dim %lld\n",
              name, k, (long long)dim, (long long)out_shape[k]);
      return false;
    }
    running *= dim;
  }
  return true;
}

template <typename T>
static int launch_qelementwise(hipStream_t stream, const void *lhs,
                               const void *rhs, void *output, int64_t total,
                               int out_rank, int kind, float M_a, float M_b,
                               int32_t lhs_zp, int32_t rhs_zp, int32_t output_zp,
                               const QElemDimsND &out_shape,
                               const QElemDimsND &lhs_stride,
                               const QElemDimsND &rhs_stride) {
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  (void)hipGetLastError();
  hipLaunchKernelGGL(qelementwise_nd_kernel<T>, dim3(grid_size),
                     dim3(kBlockSize), 0, stream, static_cast<const T *>(lhs),
                     static_cast<const T *>(rhs), static_cast<T *>(output),
                     total, out_rank, kind, M_a, M_b, lhs_zp, rhs_zp, output_zp,
                     out_shape, lhs_stride, rhs_stride);
  return 0;
}

extern "C" int hip_qelementwise(void *stream, const void *lhs, const void *rhs,
                                void *output, int64_t kind,
                                const int64_t *lhs_shape, int64_t lhs_rank,
                                const int64_t *rhs_shape, int64_t rhs_rank,
                                const int64_t *out_shape, int64_t out_rank,
                                int hip_dtype, float M_a, int64_t lhs_zp,
                                float M_b, int64_t rhs_zp, int64_t output_zp) {
  if (kind != HIP_QELEMENTWISE_ADD) {
    fprintf(stderr, "[custom_kernels] hip_qelementwise: unsupported kind=%lld\n",
            (long long)kind);
    return -1;
  }
  if (out_rank < 0 || out_rank > HIP_QELEMENTWISE_MAX_RANK || lhs_rank < 0 ||
      lhs_rank > out_rank || rhs_rank < 0 || rhs_rank > out_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_qelementwise: invalid ranks (max=%d): "
            "lhs=%lld, rhs=%lld, out=%lld\n",
            HIP_QELEMENTWISE_MAX_RANK, (long long)lhs_rank, (long long)rhs_rank,
            (long long)out_rank);
    return -1;
  }

  int64_t total = 1;
  for (int i = 0; i < out_rank; ++i)
    total *= out_shape[i];
  if (total <= 0)
    return 0;

  QElemDimsND out_shape_nd = {};
  QElemDimsND lhs_shape_nd = {}, rhs_shape_nd = {};
  QElemDimsND lhs_stride = {}, rhs_stride = {};
  for (int i = 0; i < out_rank; ++i)
    out_shape_nd.v[i] = out_shape[i];

  if (!make_broadcast_layout(lhs_shape, (int)lhs_rank, (int)out_rank, out_shape,
                             &lhs_shape_nd, &lhs_stride, "lhs"))
    return -1;
  if (!make_broadcast_layout(rhs_shape, (int)rhs_rank, (int)out_rank, out_shape,
                             &rhs_shape_nd, &rhs_stride, "rhs"))
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int k = static_cast<int>(kind);
  int32_t za = static_cast<int32_t>(lhs_zp);
  int32_t zb = static_cast<int32_t>(rhs_zp);
  int32_t zo = static_cast<int32_t>(output_zp);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qelementwise: dtype=%d kind=%d out_rank=%d "
      "total=%lld M_a=%g M_b=%g zp=(%d,%d,%d)\n",
      hip_dtype, k, (int)out_rank, (long long)total, (double)M_a, (double)M_b,
      za, zb, zo);

  int rc = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_INT8:
    rc = launch_qelementwise<int8_t>(hip_stream, lhs, rhs, output, total,
                                     (int)out_rank, k, M_a, M_b, za, zb, zo,
                                     out_shape_nd, lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_UINT8:
    rc = launch_qelementwise<uint8_t>(hip_stream, lhs, rhs, output, total,
                                      (int)out_rank, k, M_a, M_b, za, zb, zo,
                                      out_shape_nd, lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_INT16:
    rc = launch_qelementwise<int16_t>(hip_stream, lhs, rhs, output, total,
                                      (int)out_rank, k, M_a, M_b, za, zb, zo,
                                      out_shape_nd, lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_UINT16:
    rc = launch_qelementwise<uint16_t>(hip_stream, lhs, rhs, output, total,
                                       (int)out_rank, k, M_a, M_b, za, zb, zo,
                                       out_shape_nd, lhs_stride, rhs_stride);
    break;
  case HIP_DTYPE_INT32:
    rc = launch_qelementwise<int32_t>(hip_stream, lhs, rhs, output, total,
                                      (int)out_rank, k, M_a, M_b, za, zb, zo,
                                      out_shape_nd, lhs_stride, rhs_stride);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_qelementwise: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qelementwise launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
