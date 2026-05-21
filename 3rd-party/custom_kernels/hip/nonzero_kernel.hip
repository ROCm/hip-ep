/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Predicate: is the input element `non-zero` per ONNX NonZero semantics?
// For floating types this includes "not NaN" — ONNX treats NaN as non-zero
// (it is "not equal to zero" by IEEE), matching ORT CPU behaviour.
template <typename T>
__device__ __forceinline__ bool nz_is_nonzero(T v);

template <> __device__ __forceinline__ bool nz_is_nonzero<int8_t>(int8_t v) { return v != 0; }
template <> __device__ __forceinline__ bool nz_is_nonzero<uint8_t>(uint8_t v) { return v != 0; }
template <> __device__ __forceinline__ bool nz_is_nonzero<int32_t>(int32_t v) { return v != 0; }
template <> __device__ __forceinline__ bool nz_is_nonzero<int64_t>(int64_t v) { return v != 0; }
template <> __device__ __forceinline__ bool nz_is_nonzero<float>(float v) {
  // ONNX: non-zero ⇔ v != 0.0; NaN ≠ 0 ⇒ NaN counts as non-zero. Cannot
  // use `v != 0.0` directly because NaN compares unequal to itself — but
  // that's exactly what we want (NaN is "non-zero" by NaN != 0).
  return v != 0.0f;
}
template <> __device__ __forceinline__ bool nz_is_nonzero<double>(double v) {
  return v != 0.0;
}
template <> __device__ __forceinline__ bool nz_is_nonzero<__half>(__half v) {
  return __half2float(v) != 0.0f;
}

//===----------------------------------------------------------------------===//
// Pass 1: hip_nonzero_count<T>
//
// One thread per input element; bumps a single int64 atomic counter when
// the element is non-zero. Block-level shared-memory reduction first
// reduces the local count down to the block leader, which then issues a
// single atomicAdd on the global counter. This drops contention from
// num_elements atomics to grid_size atomics (e.g. for num_elements=4M and
// block=256 that's 16k atomics instead of 4M).
//===----------------------------------------------------------------------===//

template <typename T, int BS>
__global__ void hip_nonzero_count_kernel(const T *__restrict__ input,
                                         int64_t num_elements,
                                         int64_t *count_out) {
  __shared__ int64_t s_count[1];
  if (threadIdx.x == 0)
    s_count[0] = 0;
  __syncthreads();

  // Grid-stride loop so we handle num_elements > grid_size * BS too.
  int64_t local = 0;
  for (int64_t i = (int64_t)blockIdx.x * BS + threadIdx.x; i < num_elements;
       i += (int64_t)gridDim.x * BS) {
    if (nz_is_nonzero<T>(input[i]))
      local += 1;
  }
  if (local > 0)
    atomicAdd((unsigned long long *)&s_count[0],
              (unsigned long long)local);
  __syncthreads();
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long *)count_out,
              (unsigned long long)s_count[0]);
}

// Kernel that zeros a single int64 device pointer before the count pass.
// Avoids a per-call hipMemsetAsync (which is async-eligible but goes
// through a CPU helper on Windows for tiny sizes).
__global__ void hip_nonzero_zero_int64_kernel(int64_t *p) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *p = 0;
}

//===----------------------------------------------------------------------===//
// Pass 2: hip_nonzero_fill<T>
//
// Atomic-allocate per non-zero element. For each i in [0, num_elements)
// we evaluate input[i]; non-zero elements get a slot via atomicAdd on
// `atomic_idx` and then decompose i into per-axis coordinates which are
// written to output[axis * N + slot].
//
// Output layout: [rank, N] int64 — row 0 is the indices along axis 0,
// row 1 along axis 1, etc. This matches numpy.nonzero() and ONNX
// NonZero. Within each row the values are in input row-major order.
//===----------------------------------------------------------------------===//

template <typename T, int BS>
__global__ void hip_nonzero_fill_kernel(const T *__restrict__ input,
                                        int64_t num_elements, int64_t rank,
                                        int64_t N,
                                        const int64_t *__restrict__ shape,
                                        int64_t *atomic_idx, int64_t *output) {
  // Grid-stride loop, same as count pass — gives identical ordering of
  // atomicAdd calls within a single warp.
  for (int64_t i = (int64_t)blockIdx.x * BS + threadIdx.x; i < num_elements;
       i += (int64_t)gridDim.x * BS) {
    if (!nz_is_nonzero<T>(input[i]))
      continue;
    int64_t slot = atomicAdd((unsigned long long *)atomic_idx, 1ull);
    // Allocate slot; if we somehow overshoot, abort silently. This can
    // only happen when N was undersized vs. what count produced — that
    // would be a wrapper bug, not a runtime input bug.
    if (slot >= N)
      continue;
    // Decompose row-major linear index i into per-axis coordinates,
    // writing each into the corresponding output row.
    int64_t rem = i;
    for (int64_t a = rank - 1; a >= 0; --a) {
      int64_t dim_a = shape[a];
      int64_t coord = (dim_a > 0) ? (rem % dim_a) : 0;
      rem = (dim_a > 0) ? (rem / dim_a) : 0;
      output[a * N + slot] = coord;
    }
  }
}

//===----------------------------------------------------------------------===//
// Launcher dispatch
//===----------------------------------------------------------------------===//

static int dispatch_count(hipStream_t stream, const void *input,
                          int64_t num_elements, int64_t hip_dtype,
                          int64_t *count_out) {
  // Reset the counter on device. Single-block, single-thread kernel —
  // cheaper than hipMemsetAsync(0) on Windows for tiny payloads.
  hipLaunchKernelGGL(hip_nonzero_zero_int64_kernel, dim3(1), dim3(1), 0,
                     stream, count_out);

  constexpr int BS = 256;
  // Bound grid by num_elements / BS, but cap at a reasonable max so we
  // don't oversubscribe on huge inputs (the grid-stride loop handles
  // overflow). 65535 is the per-dim grid limit on HIP / RDNA3; 4096
  // blocks is a fine sweet spot for occupancy + atomicAdd contention.
  uint32_t grid = static_cast<uint32_t>(
      std::min<int64_t>(4096, (num_elements + BS - 1) / BS));
  if (grid == 0)
    grid = 1;

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<float, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const float *>(input), num_elements,
                       count_out);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<__half, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const __half *>(input), num_elements,
                       count_out);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<double, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const double *>(input), num_elements,
                       count_out);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<int32_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int32_t *>(input), num_elements,
                       count_out);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<int64_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int64_t *>(input), num_elements,
                       count_out);
    break;
  case HIP_DTYPE_INT8:
    // Also covers ONNX bool and uint8 -- the predicate is byte-wise != 0
    // so signedness doesn't matter.
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_count_kernel<int8_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int8_t *>(input), num_elements,
                       count_out);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_nonzero_count: unsupported hip_dtype=%lld\n",
            (long long)hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

static int dispatch_fill(hipStream_t stream, const void *input,
                         int64_t num_elements, int64_t hip_dtype,
                         int64_t rank, int64_t N,
                         const int64_t *shape, int64_t *atomic_idx,
                         int64_t *output) {
  // Zero the per-call atomic counter.
  hipLaunchKernelGGL(hip_nonzero_zero_int64_kernel, dim3(1), dim3(1), 0,
                     stream, atomic_idx);

  constexpr int BS = 256;
  uint32_t grid = static_cast<uint32_t>(
      std::min<int64_t>(4096, (num_elements + BS - 1) / BS));
  if (grid == 0)
    grid = 1;

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<float, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const float *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<__half, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const __half *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<double, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const double *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int32_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int32_t *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int64_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int64_t *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  case HIP_DTYPE_INT8:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int8_t, BS>),
                       dim3(grid), dim3(BS), 0, stream,
                       static_cast<const int8_t *>(input), num_elements, rank,
                       N, shape, atomic_idx, output);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_nonzero_fill: unsupported hip_dtype=%lld\n",
            (long long)hip_dtype);
    return -1;
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_nonzero_count(void *stream, const void *input,
                                 int64_t input_num_elements,
                                 int64_t input_data_type, void *count_out) {
  if (!input || !count_out) {
    fprintf(stderr, "[custom_kernels] hip_nonzero_count: null pointer\n");
    return -1;
  }
  if (input_num_elements <= 0) {
    // Still need to zero count_out so the wrapper observes 0.
    hipLaunchKernelGGL(hip_nonzero_zero_int64_kernel, dim3(1), dim3(1), 0,
                       static_cast<hipStream_t>(stream),
                       static_cast<int64_t *>(count_out));
    return static_cast<int>(hipGetLastError());
  }
  return dispatch_count(static_cast<hipStream_t>(stream), input,
                        input_num_elements, input_data_type,
                        static_cast<int64_t *>(count_out));
}

extern "C" int hip_nonzero_fill(void *stream, const void *input,
                                int64_t input_num_elements,
                                int64_t input_data_type, int64_t rank,
                                int64_t N, const void *input_shape_dev,
                                void *atomic_idx, void *output) {
  if (!input || !atomic_idx || !output) {
    fprintf(stderr, "[custom_kernels] hip_nonzero_fill: null pointer\n");
    return -1;
  }
  if (N <= 0 || rank <= 0) {
    // Nothing to write; still zero the atomic counter so subsequent
    // callers don't observe stale values if they reuse the buffer.
    hipLaunchKernelGGL(hip_nonzero_zero_int64_kernel, dim3(1), dim3(1), 0,
                       static_cast<hipStream_t>(stream),
                       static_cast<int64_t *>(atomic_idx));
    return static_cast<int>(hipGetLastError());
  }
  if (!input_shape_dev) {
    fprintf(stderr,
            "[custom_kernels] hip_nonzero_fill: null input_shape_dev (rank=%lld "
            "requires shape)\n",
            (long long)rank);
    return -1;
  }
  return dispatch_fill(static_cast<hipStream_t>(stream), input,
                       input_num_elements, input_data_type, rank, N,
                       static_cast<const int64_t *>(input_shape_dev),
                       static_cast<int64_t *>(atomic_idx),
                       static_cast<int64_t *>(output));
}
