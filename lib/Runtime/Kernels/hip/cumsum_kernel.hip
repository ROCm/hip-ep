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
 * Many short slices (vision models: large outer*inner, small axis): one
 * thread per slice scans `axis_size` elements serially with stride `inner`.
 * Few long slices (e.g. positions over a decode attention mask, one slice of
 * total_sequence_length): one block per slice walks the axis in tiles,
 * block-scanning each tile and carrying its total into the next; the serial
 * path is latency-bound when there are few slices.
 *
 * Attribute handling:
 *   exclusive=0 (default): y[k] = sum(x[0..k])
 *   exclusive=1          : y[k] = sum(x[0..k-1]), y[0] = 0
 *   reverse=0 (default)  : forward scan
 *   reverse=1            : scan from the end (y[k] = sum(x[k..N-1]) etc.)
 *
 * Ported from onnxruntime/core/providers/cuda/math/cumsum_impl.cu @ v1.22.2.
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

constexpr int kScanThreads = 256;
constexpr int kScanItems = 8;
constexpr int64_t kScanTile = kScanThreads * kScanItems;

template <typename T, typename ACC>
__global__ void __launch_bounds__(kScanThreads)
    hip_cumsum_block_kernel(const T *__restrict__ x, T *__restrict__ y,
                            int64_t axis_size, int64_t inner, bool exclusive,
                            bool reverse) {
  __shared__ ACC partial[kScanThreads];
  const int64_t slice = blockIdx.x;
  const int64_t base = (slice / inner) * axis_size * inner + slice % inner;
  const T *xs = x + base;
  T *ys = y + base;
  const int t = threadIdx.x;
  auto at = [&](int64_t k) {
    return (reverse ? axis_size - 1 - k : k) * inner;
  };

  ACC carry = ACC(0);
  for (int64_t tile = 0; tile < axis_size; tile += kScanTile) {
    const int64_t first = tile + static_cast<int64_t>(t) * kScanItems;
    ACC v[kScanItems];
    ACC sum = ACC(0);
#pragma unroll
    for (int j = 0; j < kScanItems; ++j) {
      v[j] = first + j < axis_size ? static_cast<ACC>(xs[at(first + j)])
                                   : ACC(0);
      sum += v[j];
    }
    partial[t] = sum;
    __syncthreads();
    for (int off = 1; off < kScanThreads; off <<= 1) {
      const ACC p = t >= off ? partial[t - off] : ACC(0);
      __syncthreads();
      partial[t] += p;
      __syncthreads();
    }
    ACC acc = carry + (t > 0 ? partial[t - 1] : ACC(0));
    carry += partial[kScanThreads - 1];
#pragma unroll
    for (int j = 0; j < kScanItems; ++j) {
      if (first + j >= axis_size) {
        break;
      }
      if (exclusive) {
        ys[at(first + j)] = static_cast<T>(acc);
        acc += v[j];
      } else {
        acc += v[j];
        ys[at(first + j)] = static_cast<T>(acc);
      }
    }
    __syncthreads();
  }
}

constexpr int kSerialBlockSize = 256;

static unsigned serial_grid(int64_t num_slices) {
  return static_cast<unsigned>((num_slices + kSerialBlockSize - 1) /
                               kSerialBlockSize);
}

static void launch_cumsum_serial(hipStream_t stream, const __half *x,
                                 __half *y, int64_t outer, int64_t axis_size,
                                 int64_t inner, bool exclusive, bool reverse) {
  hipLaunchKernelGGL(hip_cumsum_kernel_f16, dim3(serial_grid(outer * inner)),
                     dim3(kSerialBlockSize), 0, stream, x, y, outer,
                     axis_size, inner, exclusive, reverse);
}

template <typename T>
static void launch_cumsum_serial(hipStream_t stream, const T *x, T *y,
                                 int64_t outer, int64_t axis_size,
                                 int64_t inner, bool exclusive, bool reverse) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<T>),
                     dim3(serial_grid(outer * inner)), dim3(kSerialBlockSize),
                     0, stream, x, y, outer, axis_size, inner, exclusive,
                     reverse);
}

template <typename T, typename ACC>
static void launch_cumsum(hipStream_t stream, const void *x, void *y,
                          int64_t outer, int64_t axis_size, int64_t inner,
                          bool exclusive, bool reverse) {
  const int64_t num_slices = outer * inner;
  constexpr int64_t kBlockScanMaxSlices = 2048;
  if (axis_size >= kScanThreads && num_slices <= kBlockScanMaxSlices) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_block_kernel<T, ACC>),
                       dim3(static_cast<unsigned>(num_slices)),
                       dim3(kScanThreads), 0, stream,
                       static_cast<const T *>(x), static_cast<T *>(y),
                       axis_size, inner, exclusive, reverse);
    return;
  }
  launch_cumsum_serial(stream, static_cast<const T *>(x), static_cast<T *>(y),
                       outer, axis_size, inner, exclusive, reverse);
}

extern "C" int hip_cumsum(void *stream, const void *x, void *y,
                          int64_t outer, int64_t axis_size, int64_t inner,
                          int hip_dtype, int exclusive, int reverse) {
  if (outer <= 0 || axis_size <= 0 || inner <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  bool excl = exclusive != 0;
  bool rev = reverse != 0;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_cumsum: dtype=%d, outer=%lld, axis=%lld, "
      "inner=%lld, exclusive=%d, reverse=%d\n",
      hip_dtype, (long long)outer, (long long)axis_size, (long long)inner,
      exclusive, reverse);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    launch_cumsum<__half, float>(hip_stream, x, y, outer, axis_size, inner,
                                 excl, rev);
    break;
  case HIP_DTYPE_FLOAT32:
    launch_cumsum<float, float>(hip_stream, x, y, outer, axis_size, inner,
                                excl, rev);
    break;
  case HIP_DTYPE_INT32:
    launch_cumsum<int32_t, int32_t>(hip_stream, x, y, outer, axis_size, inner,
                                    excl, rev);
    break;
  case HIP_DTYPE_INT64:
    launch_cumsum<int64_t, int64_t>(hip_stream, x, y, outer, axis_size, inner,
                                    excl, rev);
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
