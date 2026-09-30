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
#include <cstdlib>
#include <string>

#define HIP_QELEMENTWISE_MAX_RANK 8
#define HIP_QELEMENTWISE_ADD 0
#define HIP_QELEMENTWISE_MUL 1

struct QElemDimsND {
  int64_t v[HIP_QELEMENTWISE_MAX_RANK];
};

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only. Device-side bounds therefore come
// from the <cstdint> macros through explicitly annotated helpers.
template <typename T>
struct QElemBounds;

template <>
struct QElemBounds<int8_t> {
  __device__ static float lowest() { return static_cast<float>(INT8_MIN); }
  __device__ static float highest() { return static_cast<float>(INT8_MAX); }
};

template <>
struct QElemBounds<uint8_t> {
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT8_MAX); }
};

template <>
struct QElemBounds<int16_t> {
  __device__ static float lowest() { return static_cast<float>(INT16_MIN); }
  __device__ static float highest() { return static_cast<float>(INT16_MAX); }
};

template <>
struct QElemBounds<uint16_t> {
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT16_MAX); }
};

template <>
struct QElemBounds<int32_t> {
  __device__ static float lowest() { return static_cast<float>(INT32_MIN); }
  // INT32_MAX is not representable in f32 and rounds up to 2^31, which would
  // overflow the cast back to int32_t. Clamp to the largest f32 below 2^31.
  __device__ static float highest() { return 2147483520.0f; }
};

template <typename T>
__device__ inline T q_saturate(float v) {
  return static_cast<T>(
      fminf(QElemBounds<T>::highest(), fmaxf(QElemBounds<T>::lowest(), v)));
}

// The one place the requantized arithmetic lives, so the contiguous fast path
// below cannot drift from the generic ND path it shortcuts.
template <typename T>
__device__ inline T qelem_apply(T lhs_v, T rhs_v, int kind, float M_a, float M_b,
                                int32_t lhs_zp, int32_t rhs_zp,
                                int32_t output_zp) {
  float da = static_cast<float>(lhs_v) - static_cast<float>(lhs_zp);
  float db = static_cast<float>(rhs_v) - static_cast<float>(rhs_zp);
  float acc = 0.f;
  if (kind == HIP_QELEMENTWISE_ADD)
    acc = M_a * da + M_b * db;
  else if (kind == HIP_QELEMENTWISE_MUL)
    // M_b is unused here: the host folded s_a * s_b / s_out into M_a, so the
    // product of the integer differences takes a single rescale.
    acc = M_a * (da * db);
  return q_saturate<T>(rintf(acc) + static_cast<float>(output_zp));
}

// Four elements per thread is what lets the fast path issue the same width of
// global access the neighbouring quantize/dequantize kernels already use.
template <typename T> struct QElemVec4;
template <> struct QElemVec4<int8_t> { using type = char4; };
template <> struct QElemVec4<uint8_t> { using type = uchar4; };
template <> struct QElemVec4<int16_t> { using type = short4; };
template <> struct QElemVec4<uint16_t> { using type = ushort4; };
template <> struct QElemVec4<int32_t> { using type = int4; };

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

  output[idx] = qelem_apply<T>(lhs[lhs_idx], rhs[rhs_idx], kind, M_a, M_b,
                               lhs_zp, rhs_zp, output_zp);
}

// Neither input broadcasts, so lhs, rhs and output all share the output's dense
// layout and the ND kernel's per-element rank-deep divide/modulo walk collapses
// to the linear id. Threads below vec_count take four elements through a single
// vector access; the handful past it mop up a total that is not a multiple of
// four.
template <typename T>
__global__ void qelementwise_contig_kernel(
    const T *__restrict__ lhs, const T *__restrict__ rhs, T *__restrict__ output,
    int64_t total_elements, int64_t vec_count, int kind, float M_a, float M_b,
    int32_t lhs_zp, int32_t rhs_zp, int32_t output_zp) {
  using Vec = typename QElemVec4<T>::type;
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < vec_count) {
    const Vec a = reinterpret_cast<const Vec *>(lhs)[tid];
    const Vec b = reinterpret_cast<const Vec *>(rhs)[tid];
    Vec o;
    o.x = qelem_apply<T>(a.x, b.x, kind, M_a, M_b, lhs_zp, rhs_zp, output_zp);
    o.y = qelem_apply<T>(a.y, b.y, kind, M_a, M_b, lhs_zp, rhs_zp, output_zp);
    o.z = qelem_apply<T>(a.z, b.z, kind, M_a, M_b, lhs_zp, rhs_zp, output_zp);
    o.w = qelem_apply<T>(a.w, b.w, kind, M_a, M_b, lhs_zp, rhs_zp, output_zp);
    reinterpret_cast<Vec *>(output)[tid] = o;
    return;
  }

  const int64_t idx = (vec_count << 2) + (tid - vec_count);
  if (idx < total_elements)
    output[idx] = qelem_apply<T>(lhs[idx], rhs[idx], kind, M_a, M_b, lhs_zp,
                                 rhs_zp, output_zp);
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

// make_broadcast_layout emits stride 0 for every axis of extent 1 and the
// running dense stride otherwise, so matching the dense strides is the same as
// saying the operand covers the output one-to-one. Axes the output itself has
// collapsed to 1 are exempt: their coordinate is always 0, so whatever stride
// they carry never reaches the address. Skipping them matters because the
// model's activations are rank-3 with a leading batch of 1, which would
// otherwise disqualify every tensor in the graph.
static bool is_dense_layout(int out_rank, const QElemDimsND &out_shape,
                            const QElemDimsND &stride) {
  int64_t running = 1;
  for (int k = out_rank - 1; k >= 0; --k) {
    if (out_shape.v[k] != 1 && stride.v[k] != running)
      return false;
    running *= out_shape.v[k];
  }
  return true;
}

// Lets a run be pinned to the generic path so the fast path can be A/B'd in
// one session, where thermal drift would otherwise swamp the difference.
static bool qelementwise_contig_enabled() {
  static const bool enabled = [] {
    std::string raw;
#ifdef _WIN32
    char *value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, "HIPDNN_QELEMENTWISE_CONTIG") == 0 && value) {
      raw = value;
      free(value);
    }
#else
    if (const char *value = getenv("HIPDNN_QELEMENTWISE_CONTIG"))
      raw = value;
#endif
    return !(raw == "0" || raw == "off" || raw == "OFF");
  }();
  return enabled;
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

  using Vec = typename QElemVec4<T>::type;
  const bool dense = qelementwise_contig_enabled() &&
                     is_dense_layout(out_rank, out_shape, lhs_stride) &&
                     is_dense_layout(out_rank, out_shape, rhs_stride);
  // A vector access that straddles its natural alignment faults, and nothing
  // guarantees a tensor view starts on one, so the width has to be earned.
  const bool aligned = (reinterpret_cast<uintptr_t>(lhs) % sizeof(Vec)) == 0 &&
                       (reinterpret_cast<uintptr_t>(rhs) % sizeof(Vec)) == 0 &&
                       (reinterpret_cast<uintptr_t>(output) % sizeof(Vec)) == 0;

  if (dense && aligned) {
    const int64_t vec_count = total >> 2;
    const int64_t threads = vec_count + (total - (vec_count << 2));
    int grid_size = static_cast<int>((threads + kBlockSize - 1) / kBlockSize);
    (void)hipGetLastError();
    hipLaunchKernelGGL(qelementwise_contig_kernel<T>, dim3(grid_size),
                       dim3(kBlockSize), 0, stream,
                       static_cast<const T *>(lhs), static_cast<const T *>(rhs),
                       static_cast<T *>(output), total, vec_count, kind, M_a,
                       M_b, lhs_zp, rhs_zp, output_zp);
    return 0;
  }

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
  if (kind != HIP_QELEMENTWISE_ADD && kind != HIP_QELEMENTWISE_MUL) {
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
