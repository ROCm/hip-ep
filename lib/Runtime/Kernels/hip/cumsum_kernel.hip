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
 * Two strategies, chosen on the shape:
 *
 *   slice-parallel (many slices): one thread per (outer, inner) slice, each
 *     sequentially scanning `axis_size` elements with stride `inner`.
 * O(axis_size) dependent adds per slice, but maximally parallel across slices,
 * which is the right shape for the way CumSum is used in vision models (large
 *     outer*inner, small axis).
 *
 *   slice-cooperative (few slices, long axis): one block per slice, scanning
 * the axis in tiles through LDS. The slice-parallel form only wins while there
 *     are enough slices to fill the machine; below that its per-thread serial
 *     latency is fully exposed. A decoder's attention mask is the pathological
 *     case -- [1, context] scanned on axis 1 collapses to outer = inner = 1, so
 *     ONE thread serially scanned the entire context. An SQTT capture of
 *     Gemma-4 26B-A4B on gfx1151 put that at 61 us at 2 240 and 501 us at
 *     16 241, growing linearly with context, for an op the model issues once
 *     per decode token.
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

// One slice, scanned cooperatively by a whole block in BLOCK-sized tiles.
//
// Each tile is scanned in LDS with Hillis-Steele (log2(BLOCK) steps) and the
// running total is carried across tiles, so the dependent-step count per slice
// falls from axis_size to (axis_size / BLOCK) * log2(BLOCK) -- at BLOCK=256 and
// a 16 241-long axis, from 16 241 to roughly 512.
//
// Numerics are unchanged for the integer types. For float the summation order
// within a tile differs from the serial scan, which is the usual reassociation
// a parallel scan implies.
template <typename T, typename ACC, int BLOCK>
__device__ inline void
cumsum_slice_cooperative(const T *__restrict__ x_slice, T *__restrict__ y_slice,
                         int64_t axis_size, int64_t inner, bool exclusive,
                         bool reverse) {
  __shared__ ACC tile[BLOCK];
  const int tid = static_cast<int>(threadIdx.x);
  ACC carry = ACC(0);

  for (int64_t tile_base = 0; tile_base < axis_size; tile_base += BLOCK) {
    const int64_t k = tile_base + tid;
    const bool active = (k < axis_size);
    // A reverse scan is the same scan over the slice read from the far end, so
    // mapping logical position k to physical element axis_size-1-k lets both
    // directions share everything below.
    const int64_t phys = reverse ? (axis_size - 1 - k) : k;
    const ACC v = active ? static_cast<ACC>(x_slice[phys * inner]) : ACC(0);

    tile[tid] = v;
    __syncthreads();
    for (int off = 1; off < BLOCK; off <<= 1) {
      const ACC add = (tid >= off) ? tile[tid - off] : ACC(0);
      __syncthreads();
      tile[tid] += add;
      __syncthreads();
    }

    // tile[] now holds the tile-local inclusive scan, so tile[tid - 1] is this
    // thread's exclusive value exactly. Taking it from the array rather than
    // subtracting v back off the inclusive result keeps the float path free of
    // the cancellation that inclusive-minus-self would introduce.
    if (active) {
      const ACC prefix =
          exclusive ? (tid == 0 ? ACC(0) : tile[tid - 1]) : tile[tid];
      y_slice[phys * inner] = static_cast<T>(carry + prefix);
    }
    // Inactive lanes contributed zero, so this stays correct on the ragged
    // tile.
    carry += tile[BLOCK - 1];
    // Ordered against the next iteration's write to tile[tid], not the reads
    // above, which the scan loop already synchronised.
    __syncthreads();
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

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void hip_cumsum_long_axis_kernel_f16(
    const __half *__restrict__ x, __half *__restrict__ y, int64_t outer,
    int64_t axis_size, int64_t inner, bool exclusive, bool reverse) {
  int64_t slice = static_cast<int64_t>(blockIdx.x);
  if (slice >= outer * inner)
    return;
  int64_t o = slice / inner;
  int64_t i = slice % inner;
  int64_t base = o * axis_size * inner + i;
  cumsum_slice_cooperative<__half, float, BLOCK>(x + base, y + base, axis_size,
                                                 inner, exclusive, reverse);
}

template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void hip_cumsum_long_axis_kernel(
    const T *__restrict__ x, T *__restrict__ y, int64_t outer,
    int64_t axis_size, int64_t inner, bool exclusive, bool reverse) {
  int64_t slice = static_cast<int64_t>(blockIdx.x);
  if (slice >= outer * inner)
    return;
  int64_t o = slice / inner;
  int64_t i = slice % inner;
  int64_t base = o * axis_size * inner + i;
  cumsum_slice_cooperative<T, T, BLOCK>(x + base, y + base, axis_size, inner,
                                        exclusive, reverse);
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

  // Selection is on the axis alone. Above the floor the cooperative kernel wins
  // regardless of slice count -- it stays ahead even when there are thousands
  // of slices to spread across the machine, because each of those slices still
  // pays the full serial latency of its own axis -- so a slice-count term would
  // only cost throughput.
  //
  // The floor itself is conservative rather than a measured crossover: a
  // 256-thread block scanning a short axis leaves most of its lanes idle and
  // pays the syncs anyway, and the vision shape this kernel was written for
  // (4096x16) is already fast on the existing path, which is not worth risking
  // to chase.
  constexpr int64_t kMinAxisForCooperative = 512;
  // One block per slice, so the slice count has to be expressible as a grid.
  constexpr int64_t kMaxCooperativeSlices = 1 << 30;
  const bool cooperative = axis_size >= kMinAxisForCooperative &&
                           num_slices <= kMaxCooperativeSlices;
  const int coop_grid = static_cast<int>(num_slices);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_cumsum: dtype=%d, outer=%lld, axis=%lld, "
      "inner=%lld, exclusive=%d, reverse=%d, path=%s\n",
      hip_dtype, (long long)outer, (long long)axis_size, (long long)inner,
      exclusive, reverse, cooperative ? "cooperative" : "slice-parallel");

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    if (cooperative)
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_long_axis_kernel_f16<kBlockSize>),
          dim3(coop_grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const __half *>(x), static_cast<__half *>(y), outer,
          axis_size, inner, excl, rev);
    else
      hipLaunchKernelGGL(
          hip_cumsum_kernel_f16, dim3(grid_size), dim3(kBlockSize), 0,
          hip_stream, static_cast<const __half *>(x), static_cast<__half *>(y),
          outer, axis_size, inner, excl, rev);
    break;
  case HIP_DTYPE_FLOAT32:
    if (cooperative)
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_long_axis_kernel<float, kBlockSize>),
          dim3(coop_grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const float *>(x), static_cast<float *>(y), outer,
          axis_size, inner, excl, rev);
    else
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_cumsum_kernel<float>),
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const float *>(x), static_cast<float *>(y),
                         outer, axis_size, inner, excl, rev);
    break;
  case HIP_DTYPE_INT32:
    if (cooperative)
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_long_axis_kernel<int32_t, kBlockSize>),
          dim3(coop_grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int32_t *>(x), static_cast<int32_t *>(y), outer,
          axis_size, inner, excl, rev);
    else
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_kernel<int32_t>), dim3(grid_size),
          dim3(kBlockSize), 0, hip_stream, static_cast<const int32_t *>(x),
          static_cast<int32_t *>(y), outer, axis_size, inner, excl, rev);
    break;
  case HIP_DTYPE_INT64:
    if (cooperative)
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_long_axis_kernel<int64_t, kBlockSize>),
          dim3(coop_grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int64_t *>(x), static_cast<int64_t *>(y), outer,
          axis_size, inner, excl, rev);
    else
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_cumsum_kernel<int64_t>), dim3(grid_size),
          dim3(kBlockSize), 0, hip_stream, static_cast<const int64_t *>(x),
          static_cast<int64_t *>(y), outer, axis_size, inner, excl, rev);
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
