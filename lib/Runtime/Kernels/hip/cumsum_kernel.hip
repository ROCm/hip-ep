/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * CumSum: y = cumulative-sum of x along `axis`.
 *
 * Layout view: collapse the input as `[outer, axis_size, inner]` where
 *   outer     = product(shape[:axis])
 *   axis_size = shape[axis]
 *   inner     = product(shape[axis+1:])
 *
 * One thread per (outer, inner) slice. Each thread sequentially scans
 * `axis_size` elements with stride `inner`. This is O(axis_size) per slice
 * but maximally parallel across slices, which is the right shape for the
 * way CumSum is used in vision models (large outer*inner, small axis).
 *
 * Attribute handling:
 *   exclusive=0 (default): y[k] = sum(x[0..k])
 *   exclusive=1          : y[k] = sum(x[0..k-1]), y[0] = 0
 *   reverse=0 (default)  : forward scan
 *   reverse=1            : scan from the end (y[k] = sum(x[k..N-1]) etc.)
 *
 * Ported from onnxruntime/core/providers/cuda/math/cumsum_impl.cu @ v1.22.2.
 * ORT uses a per-block Kogge-Stone style scan; we use a per-thread serial
 * scan -- simpler, identical numerics, and faster when slices >> threads-
 * per-block which is the common case here.
 *
 * Before: wrap_cumsum D2H'd the axis scalar and synchronized so the host
 *         could form outer/axis_size/inner and size the grid exactly.
 * After:  no host wait. Grid is num_elements (an upper bound on the
 *         slice count). Thread 0 loads axis, splits data_shape in
 *         shared memory; extra threads return. K > rank sets the
 *         session device-error flag.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kCumSumMaxRank = 8;

struct CumSumGeom {
  int64_t shape[kCumSumMaxRank];
  int rank;
};

struct CumSumResolved {
  int64_t outer;
  int64_t axis_size;
  int64_t inner;
  int error;
};

__device__ void resolve_cumsum_axis(const void *__restrict__ axis,
                                    int axis_elem_bytes, const CumSumGeom &g,
                                    CumSumResolved *out) {
  out->error = 0;
  out->outer = 0;
  out->axis_size = 0;
  out->inner = 0;
  int64_t axis_value = 0;
  if (axis_elem_bytes == 4)
    axis_value = static_cast<int64_t>(*static_cast<const int32_t *>(axis));
  else
    axis_value = *static_cast<const int64_t *>(axis);
  if (axis_value < 0)
    axis_value += g.rank;
  if (axis_value < 0 || axis_value >= g.rank) {
    out->error = -1;
    return;
  }
  int64_t outer = 1;
  for (int64_t d = 0; d < axis_value; ++d)
    outer *= g.shape[d];
  int64_t inner = 1;
  for (int64_t d = axis_value + 1; d < g.rank; ++d)
    inner *= g.shape[d];
  out->outer = outer;
  out->axis_size = g.shape[axis_value];
  out->inner = inner;
}

template <typename T, typename ACC>
__device__ inline void cumsum_one_slice(const T *__restrict__ x_slice,
                                        T *__restrict__ y_slice,
                                        int64_t axis_size, int64_t inner,
                                        bool exclusive, bool reverse) {
  ACC acc = ACC(0);
  if (!reverse) {
    if (!exclusive) {
      for (int64_t k = 0; k < axis_size; ++k) {
        acc += static_cast<ACC>(x_slice[k * inner]);
        y_slice[k * inner] = static_cast<T>(acc);
      }
    } else {
      for (int64_t k = 0; k < axis_size; ++k) {
        y_slice[k * inner] = static_cast<T>(acc);
        acc += static_cast<ACC>(x_slice[k * inner]);
      }
    }
  } else {
    if (!exclusive) {
      for (int64_t k = axis_size - 1; k >= 0; --k) {
        acc += static_cast<ACC>(x_slice[k * inner]);
        y_slice[k * inner] = static_cast<T>(acc);
      }
    } else {
      for (int64_t k = axis_size - 1; k >= 0; --k) {
        y_slice[k * inner] = static_cast<T>(acc);
        acc += static_cast<ACC>(x_slice[k * inner]);
      }
    }
  }
}

template <typename T, typename ACC>
__global__ void hip_cumsum_kernel(const T *__restrict__ x, T *__restrict__ y,
                                  CumSumGeom geom, const void *__restrict__ axis,
                                  int axis_elem_bytes, bool exclusive,
                                  bool reverse, int64_t num_elements,
                                  int *device_error_flag) {
  __shared__ CumSumResolved resolved;
  if (threadIdx.x == 0)
    resolve_cumsum_axis(axis, axis_elem_bytes, geom, &resolved);
  __syncthreads();

  if (resolved.error && device_error_flag && threadIdx.x == 0 &&
      blockIdx.x == 0)
    *device_error_flag = -1;
  if (resolved.error || resolved.outer <= 0 || resolved.axis_size <= 0 ||
      resolved.inner <= 0)
    return;

  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t num_slices = resolved.outer * resolved.inner;
  if (tid >= num_slices || tid >= num_elements)
    return;
  int64_t o = tid / resolved.inner;
  int64_t i = tid % resolved.inner;
  int64_t base = o * resolved.axis_size * resolved.inner + i;
  cumsum_one_slice<T, ACC>(x + base, y + base, resolved.axis_size,
                           resolved.inner, exclusive, reverse);
}

} // namespace

extern "C" int hip_cumsum(void *stream, const void *x, void *y,
                          const int64_t *data_shape, int64_t data_rank,
                          int64_t num_elements, const void *axis,
                          int axis_elem_bytes, int hip_dtype, int exclusive,
                          int reverse, void *device_error_flag) {
  if (!x || !y || !data_shape || !axis)
    return -1;
  if (data_rank <= 0 || data_rank > kCumSumMaxRank)
    return -1;
  if (axis_elem_bytes != 4 && axis_elem_bytes != 8)
    return -1;
  if (num_elements <= 0)
    return 0;

  CumSumGeom geom{};
  geom.rank = static_cast<int>(data_rank);
  for (int64_t d = 0; d < data_rank; ++d)
    geom.shape[d] = data_shape[d];

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  bool excl = exclusive != 0;
  bool rev = reverse != 0;
  int *err_flag = static_cast<int *>(device_error_flag);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_cumsum: dtype=%d, rank=%lld, num=%lld, "
      "exclusive=%d, reverse=%d\n",
      hip_dtype, (long long)data_rank, (long long)num_elements, exclusive,
      reverse);

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<__half, float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(x),
                       static_cast<__half *>(y), geom, axis, axis_elem_bytes,
                       excl, rev, num_elements, err_flag);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<float, float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(x),
                       static_cast<float *>(y), geom, axis, axis_elem_bytes,
                       excl, rev, num_elements, err_flag);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int32_t, int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(x),
                       static_cast<int32_t *>(y), geom, axis, axis_elem_bytes,
                       excl, rev, num_elements, err_flag);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int64_t, int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(x),
                       static_cast<int64_t *>(y), geom, axis, axis_elem_bytes,
                       excl, rev, num_elements, err_flag);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_cumsum: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_cumsum launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
