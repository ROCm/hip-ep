/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip.arg_max: one i64 index per slice along `axis`.
//
// Values are compared in their native type. A 4-byte element is not assumed
// to be float: the text-encoder ArgMax is i32. select_last_index == 0 keeps
// the first maximum; select_last_index != 0 keeps the last.
//
// The output buffer is the contiguous product of every input dim except
// `axis` (keepdims only inserts a size-1 dim, which does not change that
// count). Dynamic sizes arrive in `data_shape`.

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr int kMaxRank = 8;

struct ArgMaxGeom {
  int64_t rank;
  int64_t axis;
  int64_t in_shape[kMaxRank];
  int64_t in_stride[kMaxRank];
  int64_t num_rows;
  int64_t axis_size;
};

template <typename T>
__device__ bool argmax_take(T candidate, T best, int select_last) {
  return select_last ? (candidate >= best) : (candidate > best);
}

// Half types have no reliable integer-style compare. Order them as float.
// i32/i64 stay on the template above so a 4-byte int is not read as float.
__device__ bool argmax_take(__half candidate, __half best, int select_last) {
  float c = __half2float(candidate);
  float b = __half2float(best);
  return select_last ? (c >= b) : (c > b);
}

__device__ bool argmax_take(hip_bfloat16 candidate, hip_bfloat16 best,
                            int select_last) {
  float c = static_cast<float>(candidate);
  float b = static_cast<float>(best);
  return select_last ? (c >= b) : (c > b);
}

template <typename T>
__device__ void argmax_row(const T *__restrict__ data,
                           int64_t *__restrict__ indices, ArgMaxGeom geom,
                           int64_t row_id, int select_last) {
  int64_t coords[kMaxRank] = {0};
  int64_t rem = row_id;
  for (int64_t d = geom.rank - 1; d >= 0; --d) {
    if (d == geom.axis)
      continue;
    int64_t dim = geom.in_shape[d];
    coords[d] = (dim > 0) ? (rem % dim) : 0;
    if (dim > 0)
      rem /= dim;
  }

  int64_t base = 0;
  for (int64_t d = 0; d < geom.rank; ++d) {
    if (d == geom.axis)
      continue;
    base += coords[d] * geom.in_stride[d];
  }

  if (geom.axis_size <= 0)
    return;

  int64_t best_idx = 0;
  T best = data[base];
  int64_t axis_stride = geom.in_stride[geom.axis];
  for (int64_t i = 1; i < geom.axis_size; ++i) {
    T candidate = data[base + i * axis_stride];
    if (argmax_take(candidate, best, select_last)) {
      best = candidate;
      best_idx = i;
    }
  }
  indices[row_id] = best_idx;
}

template <typename T>
__global__ void arg_max_kernel(const T *__restrict__ data,
                               int64_t *__restrict__ indices, ArgMaxGeom geom,
                               int select_last) {
  int64_t row_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_id >= geom.num_rows)
    return;
  argmax_row(data, indices, geom, row_id, select_last);
}

template <typename T>
static int launch_arg_max(void *stream, const void *data, void *indices,
                          const ArgMaxGeom &geom, int select_last) {
  if (geom.num_rows <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((geom.num_rows + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(arg_max_kernel<T>), dim3(grid_size),
                     dim3(kBlockSize), 0, hip_stream,
                     static_cast<const T *>(data),
                     static_cast<int64_t *>(indices), geom, select_last);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] arg_max_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_arg_max(void *stream, const void *data, void *indices,
                           int64_t axis, int64_t select_last_index,
                           int64_t rank, const int64_t *data_shape,
                           int hip_dtype) {
  if (!data || !indices || !data_shape)
    return -1;
  if (rank <= 0 || rank > kMaxRank)
    return -1;

  ArgMaxGeom geom{};
  geom.rank = rank;
  geom.axis = axis;
  if (geom.axis < 0)
    geom.axis += rank;
  if (geom.axis < 0 || geom.axis >= rank)
    return -1;

  for (int64_t i = 0; i < rank; ++i) {
    if (data_shape[i] < 0)
      return -1;
    geom.in_shape[i] = data_shape[i];
  }

  geom.in_stride[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i)
    geom.in_stride[i] = geom.in_stride[i + 1] * geom.in_shape[i + 1];

  geom.axis_size = geom.in_shape[geom.axis];
  // A zero-length axis still has one output index per other dimension.
  // Launching and returning without a store leaves that memory uninitialized
  // while hip_arg_max reports success. ONNX does not define ArgMax of an
  // empty axis; fail here so wrap_arg_max propagates the error.
  if (geom.axis_size <= 0) {
    fprintf(stderr, "[custom_kernels] hip_arg_max: empty reduction axis\n");
    return -1;
  }

  geom.num_rows = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == geom.axis)
      continue;
    geom.num_rows *= geom.in_shape[i];
  }
  if (geom.num_rows > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    fprintf(stderr, "[custom_kernels] hip_arg_max: row count %lld exceeds int\n",
            (long long)geom.num_rows);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_arg_max: axis=%lld, rank=%lld, rows=%lld, "
      "axis_size=%lld, select_last=%lld, dtype=%d\n",
      (long long)geom.axis, (long long)rank, (long long)geom.num_rows,
      (long long)geom.axis_size, (long long)select_last_index, hip_dtype);

  int select_last = select_last_index != 0 ? 1 : 0;
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    return launch_arg_max<__half>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_BFLOAT16:
    return launch_arg_max<hip_bfloat16>(stream, data, indices, geom,
                                        select_last);
  case HIP_DTYPE_FLOAT32:
    return launch_arg_max<float>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_FLOAT64:
    return launch_arg_max<double>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_INT8:
    return launch_arg_max<int8_t>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_UINT8:
    return launch_arg_max<uint8_t>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_INT16:
    return launch_arg_max<int16_t>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_UINT16:
    return launch_arg_max<uint16_t>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_INT32:
    return launch_arg_max<int32_t>(stream, data, indices, geom, select_last);
  case HIP_DTYPE_INT64:
    return launch_arg_max<int64_t>(stream, data, indices, geom, select_last);
  default:
    fprintf(stderr, "[custom_kernels] hip_arg_max: unsupported hip_dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
