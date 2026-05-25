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

// IMPORTANT (ONNX NonZero spec): the output indices MUST be in
// row-major input order — i.e. slot `s` corresponds to the s-th
// non-zero element encountered when scanning `input` linearly from
// index 0. The original kernel used `atomicAdd` for slot assignment
// which produced a non-deterministic order; that breaks downstream
// consumers like ScatterND where the (indices, updates) pairing is
// order-sensitive (Slice + Reshape feed `updates` in deterministic
// flat order, so a scrambled `indices` writes the right VALUES to
// the wrong POSITIONS — passes cosine ~0.99 but element-wise diff
// is huge).
//
// We use a two-pass deterministic approach:
//
//   Pass 2a (`hip_nonzero_blockscan_kernel`): block-local exclusive
//     scan over the boolean mask. Within each block of BS elements
//     we compute per-thread `local_offset` (sum of mask bits for
//     lanes < tid) and write the per-block total to `block_sums`.
//
//   Pass 2b (`hip_nonzero_blocksum_scan_kernel`): single-block
//     exclusive scan over `block_sums` to get `block_offset[b]` =
//     sum of mask bits in blocks < b.
//
//   Pass 2c (`hip_nonzero_fill_kernel`): each thread writes its
//     non-zero element's coords to `output[a * N +
//     (block_offset[blockIdx.x] + local_offset)]`. Deterministic
//     and parallel.
//
// This matches numpy.nonzero / ONNX NonZero exactly. `BS` must be
// a power of two ≤ wavefront size for the in-warp scan to work
// correctly without `__syncthreads`. We pick BS=64 (RDNA wave size).

constexpr int kNonZeroScanBS = 64;

template <typename T>
__global__ void hip_nonzero_blockscan_kernel(const T *__restrict__ input,
                                             int64_t num_elements,
                                             int32_t *__restrict__ thread_off,
                                             int32_t *__restrict__ block_sums) {
  // Each block processes BS consecutive elements (no grid-stride loop
  // here — the scan structure requires per-block locality).
  int64_t base = (int64_t)blockIdx.x * kNonZeroScanBS;
  int tid = threadIdx.x;
  int64_t idx = base + tid;
  int32_t flag = 0;
  if (idx < num_elements && nz_is_nonzero<T>(input[idx]))
    flag = 1;
  // Warp/block-local exclusive scan via shared memory (Hillis-Steele
  // style — O(log BS) passes). Works for BS=kNonZeroScanBS (=64 =
  // RDNA wave size; one wave per block keeps it simple).
  __shared__ int32_t s[kNonZeroScanBS];
  s[tid] = flag;
  __syncthreads();
  for (int offset = 1; offset < kNonZeroScanBS; offset *= 2) {
    int32_t v = (tid >= offset) ? s[tid - offset] : 0;
    __syncthreads();
    s[tid] += v;
    __syncthreads();
  }
  // s[tid] now holds inclusive scan; convert to exclusive.
  int32_t exclusive = s[tid] - flag;
  thread_off[base + tid] =
      (idx < num_elements)
          ? exclusive
          : 0; // OOB lanes write 0 — not consumed by Pass 2c.
  // Block leader publishes the block total (inclusive scan of last
  // valid lane). We use kNonZeroScanBS - 1 even if some lanes are
  // OOB because those wrote flag=0, contributing 0 to the sum.
  if (tid == kNonZeroScanBS - 1)
    block_sums[blockIdx.x] = s[tid];
}

// Single-block exclusive scan over block_sums (up to
// kNonZeroScanBS * kNonZeroScanBS = 4096 blocks per launch — covers
// num_elements up to 4096 * 64 = 262144). For larger inputs we
// fall back to a sequential CPU-side scan inside a 1-thread kernel
// — the cost is dominated by the launch overhead, not the scan
// itself.
__global__ void hip_nonzero_blocksum_scan_kernel(
    int32_t *__restrict__ block_sums, int64_t num_blocks) {
  // Single thread: deterministic, no atomics, works for any
  // num_blocks. The cost is O(num_blocks) sequential, but the kernel
  // launch overhead dominates anyway for our problem sizes
  // (num_blocks ≈ num_elements / 64 ≈ 8k for 524288 elements,
  // sequential scan = ~10 microseconds vs ~3 microseconds of launch
  // overhead).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int32_t acc = 0;
    for (int64_t b = 0; b < num_blocks; ++b) {
      int32_t v = block_sums[b];
      block_sums[b] = acc;
      acc += v;
    }
  }
}

template <typename T>
__global__ void hip_nonzero_fill_kernel(
    const T *__restrict__ input, int64_t num_elements, int64_t rank, int64_t N,
    const int64_t *__restrict__ shape,
    const int32_t *__restrict__ thread_off,
    const int32_t *__restrict__ block_offset, int64_t *output) {
  int64_t base = (int64_t)blockIdx.x * kNonZeroScanBS;
  int tid = threadIdx.x;
  int64_t i = base + tid;
  if (i >= num_elements)
    return;
  if (!nz_is_nonzero<T>(input[i]))
    return;
  int64_t slot = (int64_t)block_offset[blockIdx.x] + thread_off[i];
  if (slot >= N)
    return;
  int64_t rem = i;
  for (int64_t a = rank - 1; a >= 0; --a) {
    int64_t dim_a = shape[a];
    int64_t coord = (dim_a > 0) ? (rem % dim_a) : 0;
    rem = (dim_a > 0) ? (rem / dim_a) : 0;
    output[a * N + slot] = coord;
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

// Dispatch the three-pass deterministic fill: blockscan → blocksum
// scan → fill. `thread_off_dev` and `block_sums_dev` are scratch
// buffers the wrapper owns (sized num_elements*sizeof(int32_t) and
// num_blocks*sizeof(int32_t) respectively). `atomic_idx` is kept in
// the API for binary compatibility but unused.
static int dispatch_fill(hipStream_t stream, const void *input,
                         int64_t num_elements, int64_t hip_dtype,
                         int64_t rank, int64_t N, const int64_t *shape,
                         int64_t *atomic_idx /*unused*/,
                         int32_t *thread_off_dev, int32_t *block_sums_dev,
                         int64_t *output) {
  (void)atomic_idx;
  if (num_elements <= 0)
    return 0;

  int64_t num_blocks = (num_elements + kNonZeroScanBS - 1) / kNonZeroScanBS;

  // Pass 2a: per-block exclusive scan over the mask.
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<float>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const float *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<__half>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const __half *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<double>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const double *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<int32_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int32_t *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<int64_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int64_t *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  case HIP_DTYPE_INT8:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_blockscan_kernel<int8_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int8_t *>(input), num_elements,
                       thread_off_dev, block_sums_dev);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_nonzero_fill: unsupported hip_dtype=%lld\n",
            (long long)hip_dtype);
    return -1;
  }

  // Pass 2b: single-thread exclusive scan over block_sums in-place.
  hipLaunchKernelGGL(hip_nonzero_blocksum_scan_kernel, dim3(1), dim3(1), 0,
                     stream, block_sums_dev, num_blocks);

  // Pass 2c: parallel fill — write coords to output[a * N + slot]
  // where slot = block_offset[blockIdx.x] + thread_off[idx]. This is
  // deterministic in input row-major order.
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<float>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const float *>(input), num_elements, rank, N,
                       shape, thread_off_dev, block_sums_dev, output);
    break;
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<__half>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const __half *>(input), num_elements, rank,
                       N, shape, thread_off_dev, block_sums_dev, output);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<double>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const double *>(input), num_elements, rank,
                       N, shape, thread_off_dev, block_sums_dev, output);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int32_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int32_t *>(input), num_elements, rank,
                       N, shape, thread_off_dev, block_sums_dev, output);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int64_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int64_t *>(input), num_elements, rank,
                       N, shape, thread_off_dev, block_sums_dev, output);
    break;
  case HIP_DTYPE_INT8:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_nonzero_fill_kernel<int8_t>),
                       dim3(static_cast<uint32_t>(num_blocks)),
                       dim3(kNonZeroScanBS), 0, stream,
                       static_cast<const int8_t *>(input), num_elements, rank,
                       N, shape, thread_off_dev, block_sums_dev, output);
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

// `thread_off_dev` and `block_sums_dev` are wrapper-owned scratch
// buffers required by the deterministic two-pass scan fill. Sizes
// (caller MUST honour):
//   thread_off_dev:  input_num_elements * sizeof(int32_t)
//   block_sums_dev:  ceil(input_num_elements / 64) * sizeof(int32_t)
// `atomic_idx` is retained for source compatibility but is unused
// by the new scan-based fill — pass any non-null device pointer.
extern "C" int hip_nonzero_fill(void *stream, const void *input,
                                int64_t input_num_elements,
                                int64_t input_data_type, int64_t rank,
                                int64_t N, const void *input_shape_dev,
                                void *atomic_idx, void *thread_off_dev,
                                void *block_sums_dev, void *output) {
  if (!input || !output) {
    fprintf(stderr, "[custom_kernels] hip_nonzero_fill: null pointer\n");
    return -1;
  }
  if (N <= 0 || rank <= 0) {
    // Nothing to write — fast-exit (no-op).
    return 0;
  }
  if (!thread_off_dev || !block_sums_dev) {
    fprintf(stderr,
            "[custom_kernels] hip_nonzero_fill: null scratch buffer "
            "(thread_off=%p, block_sums=%p)\n",
            thread_off_dev, block_sums_dev);
    return -1;
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
                       static_cast<int32_t *>(thread_off_dev),
                       static_cast<int32_t *>(block_sums_dev),
                       static_cast<int64_t *>(output));
}
