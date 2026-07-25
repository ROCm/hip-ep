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
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

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

// FP16 accumulates in float -- matches reduce_sum's fp16-precision handling
// in the same file family and avoids loss for long axes.
__global__ void hip_cumsum_kernel_f16(const __half *__restrict__ x,
                                      __half *__restrict__ y, int64_t outer,
                                      int64_t axis_size, int64_t inner,
                                      bool exclusive, bool reverse) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t num_slices = outer * inner;
  if (tid >= num_slices)
    return;
  int64_t o = tid / inner;
  int64_t i = tid % inner;
  int64_t base = o * axis_size * inner + i;
  cumsum_one_slice<__half, float>(x + base, y + base, axis_size, inner,
                                  exclusive, reverse);
}

template <typename T>
__global__ void hip_cumsum_kernel(const T *__restrict__ x, T *__restrict__ y,
                                  int64_t outer, int64_t axis_size,
                                  int64_t inner, bool exclusive, bool reverse) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t num_slices = outer * inner;
  if (tid >= num_slices)
    return;
  int64_t o = tid / inner;
  int64_t i = tid % inner;
  int64_t base = o * axis_size * inner + i;
  cumsum_one_slice<T, T>(x + base, y + base, axis_size, inner, exclusive,
                         reverse);
}

extern "C" int hip_cumsum(void *stream, const void *x, void *y,
                          int64_t outer, int64_t axis_size, int64_t inner,
                          int hip_dtype, int exclusive, int reverse) {
  if (outer <= 0 || axis_size <= 0 || inner <= 0)
    return 0;

  int64_t num_slices = outer * inner;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_slices + kBlockSize - 1) / kBlockSize);

  bool excl = exclusive != 0;
  bool rev = reverse != 0;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_cumsum: dtype=%d, outer=%lld, axis=%lld, "
      "inner=%lld, exclusive=%d, reverse=%d\n",
      hip_dtype, (long long)outer, (long long)axis_size, (long long)inner,
      exclusive, reverse);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(hip_cumsum_kernel_f16, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(x),
                       static_cast<__half *>(y), outer, axis_size, inner, excl,
                       rev);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(x),
                       static_cast<float *>(y), outer, axis_size, inner, excl,
                       rev);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(x),
                       static_cast<int32_t *>(y), outer, axis_size, inner, excl,
                       rev);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(x),
                       static_cast<int64_t *>(y), outer, axis_size, inner, excl,
                       rev);
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
