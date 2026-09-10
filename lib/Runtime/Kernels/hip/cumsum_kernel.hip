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
 * `axis` is an ONNX input, so it lives on the device. Reading it back to
 * pick the launch geometry would cost a full stream drain on every call, so
 * instead per block thread 0 loads it and does the outer/axis_size/inner
 * decomposition into shared memory. The launch is sized from an
 * axis-independent upper bound on the slice count -- `total / min(shape)`,
 * since `outer * inner == total / shape[axis]` for any axis -- and the
 * kernel walks slices with a grid-stride loop, so an over-estimate costs
 * nothing beyond the threads that exit immediately.
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
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kCumsumMaxRank = 8;

struct CumsumShape {
  int64_t shape[kCumsumMaxRank];
  int rank;
};

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

// Normalizes a negative axis and collapses the shape around it. Returns
// false for an axis outside [-rank, rank), which used to abort in the host
// wrapper back when it read the value; the caller reports it through the
// shared device error flag instead.
__device__ inline bool cumsum_decompose(const CumsumShape &s, int64_t axis,
                                        int64_t &outer, int64_t &axis_size,
                                        int64_t &inner) {
  if (axis < 0)
    axis += s.rank;
  if (axis < 0 || axis >= s.rank)
    return false;

  outer = 1;
  for (int d = 0; d < static_cast<int>(axis); ++d)
    outer *= s.shape[d];
  axis_size = s.shape[static_cast<int>(axis)];
  inner = 1;
  for (int d = static_cast<int>(axis) + 1; d < s.rank; ++d)
    inner *= s.shape[d];
  return true;
}

// ACC is the accumulator type: fp16 accumulates in float -- matching
// reduce_sum's fp16-precision handling in the same file family -- to avoid
// loss for long axes.
template <typename T, typename ACC>
__global__ void hip_cumsum_kernel(const T *__restrict__ x, T *__restrict__ y,
                                  const void *__restrict__ axis_ptr,
                                  int axis_is_int64, CumsumShape s,
                                  bool exclusive, bool reverse,
                                  int *device_error_flag) {
  // Every thread must reach the barrier, so the decomposition happens before
  // any thread can drop out of the grid-stride loop.
  __shared__ int64_t s_outer;
  __shared__ int64_t s_axis_size;
  __shared__ int64_t s_inner;
  if (threadIdx.x == 0) {
    int64_t axis =
        axis_is_int64
            ? *static_cast<const int64_t *>(axis_ptr)
            : static_cast<int64_t>(*static_cast<const int32_t *>(axis_ptr));
    if (!cumsum_decompose(s, axis, s_outer, s_axis_size, s_inner)) {
      if (device_error_flag)
        *device_error_flag = -1;
      s_outer = 0;
      s_axis_size = 0;
      s_inner = 0;
    }
  }
  __syncthreads();

  const int64_t num_slices = s_outer * s_inner;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       tid < num_slices; tid += stride) {
    int64_t o = tid / s_inner;
    int64_t i = tid % s_inner;
    int64_t base = o * s_axis_size * s_inner + i;
    cumsum_one_slice<T, ACC>(x + base, y + base, s_axis_size, s_inner,
                             exclusive, reverse);
  }
}

extern "C" int hip_cumsum(void *stream, const void *x, void *y,
                          const int64_t *data_shape_host, int rank,
                          const void *axis_dev, int axis_is_int64,
                          int hip_dtype, int exclusive, int reverse,
                          void *device_error_flag) {
  if (rank <= 0 || rank > kCumsumMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_cumsum: rank=%d out of range [1, %d]\n", rank,
            kCumsumMaxRank);
    return -1;
  }
  if (!axis_dev) {
    fprintf(stderr, "[custom_kernels] hip_cumsum: null axis\n");
    return -1;
  }

  CumsumShape s{};
  s.rank = rank;
  int64_t total = 1;
  int64_t min_dim = data_shape_host[0];
  for (int d = 0; d < rank; ++d) {
    s.shape[d] = data_shape_host[d];
    total *= data_shape_host[d];
    if (data_shape_host[d] < min_dim)
      min_dim = data_shape_host[d];
  }
  if (total <= 0)
    return 0;

  // outer * inner == total / shape[axis] for whichever axis the device
  // scalar names, so the largest it can be is total / min(shape).
  const int64_t max_slices = total / min_dim;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  // Cap the grid and let the kernel stride: the bound above is loose when a
  // size-1 dimension leads the shape, and launching millions of threads that
  // would exit immediately is pure overhead.
  constexpr int64_t kMaxBlocks = 4096;
  int64_t blocks = (max_slices + kBlockSize - 1) / kBlockSize;
  if (blocks > kMaxBlocks)
    blocks = kMaxBlocks;
  int grid_size = static_cast<int>(blocks);

  bool excl = exclusive != 0;
  bool rev = reverse != 0;
  int *err_flag = static_cast<int *>(device_error_flag);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_cumsum: dtype=%d, rank=%d, total=%lld, "
      "max_slices=%lld, exclusive=%d, reverse=%d\n",
      hip_dtype, rank, (long long)total, (long long)max_slices, exclusive,
      reverse);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<__half, float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(x),
                       static_cast<__half *>(y), axis_dev, axis_is_int64, s,
                       excl, rev, err_flag);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<float, float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(x), static_cast<float *>(y),
                       axis_dev, axis_is_int64, s, excl, rev, err_flag);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int32_t, int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(x),
                       static_cast<int32_t *>(y), axis_dev, axis_is_int64, s,
                       excl, rev, err_flag);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int64_t, int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(x),
                       static_cast<int64_t *>(y), axis_dev, axis_is_int64, s,
                       excl, rev, err_flag);
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
