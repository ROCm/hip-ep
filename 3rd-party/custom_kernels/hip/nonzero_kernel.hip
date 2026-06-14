/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * NonZero kernel: single-block cooperative ordered scan.
 *
 * Output layout: [rank, capacity] of i64. Valid columns are [0, *count_ptr),
 * and ordering matches ONNX spec: row-major over the input.
 *
 * Ordered output is required so that downstream pipelines that read the
 * indices in lockstep with a sequentially-derived companion tensor (e.g.
 * `Slice(image_features, [0:M])` feeding ScatterND alongside these
 * indices) write each update to the correct target position. The older
 * atomicAdd-only design produced random ordering -- correct for cases
 * where the update value is positional-independent, but silently wrong
 * for the embedding-substitution pattern above.
 *
 * Algorithm (one block, BS threads):
 *   1. Each thread sequentially counts nonzeros in its chunk [lo, hi).
 *   2. Thread 0 does an exclusive prefix scan over the BS chunk counts
 *      (BS is small -- 1024 -- so a single-thread scan is fastest).
 *   3. Each thread re-walks its chunk and writes coords at
 *      `chunk_offset[tid] + local_running`.
 *   4. Thread 0 also writes the total count.
 *
 * Bounded to input rank <= 8.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "hip_custom_kernels.h"

static constexpr int kMaxRank = 8;
static constexpr int kNonZeroBlockSize = 1024;

// Input dimensions passed BY VALUE (copied into the kernel argument buffer),
// not via a device pointer. The dims are tiny (<= kMaxRank int64) and known on
// the host; passing them by value removes a per-call hipMalloc/hipMemcpy/
// hipFreeAsync of the dims buffer from the launcher. That per-call allocation
// was an unchecked failure point under heavy device-memory pressure (a failed
// hipMalloc left the device dims pointer garbage -> wrong row-major coordinate
// decomposition -> indices written to wrong positions), and is exactly the
// kind of hot-path device allocation the runtime is supposed to avoid.
struct NonZeroDims {
  int64_t d[kMaxRank];
};

template <typename T>
__global__ void nonzero_ordered_kernel(const T *__restrict__ input,
                                       int64_t *__restrict__ output,
                                       int *__restrict__ count_ptr,
                                       int64_t capacity, int rank,
                                       int64_t numel, NonZeroDims dims) {
  __shared__ int s_chunk_count[kNonZeroBlockSize];
  __shared__ int s_chunk_offset[kNonZeroBlockSize];

  int tid = threadIdx.x;

  // Even chunk size (round up); last thread may have a short chunk.
  int64_t chunk = (numel + kNonZeroBlockSize - 1) / kNonZeroBlockSize;
  int64_t lo = (int64_t)tid * chunk;
  int64_t hi = lo + chunk;
  if (hi > numel)
    hi = numel;
  if (lo > numel)
    lo = numel;

  // Phase 1: count nonzeros in [lo, hi).
  int my_count = 0;
  for (int64_t i = lo; i < hi; ++i) {
    if (input[i] != (T)0)
      ++my_count;
  }
  s_chunk_count[tid] = my_count;
  __syncthreads();

  // Phase 2: thread-0 exclusive scan over BS chunk counts. BS=1024 is
  // small enough that single-thread shared-memory scan beats parallel
  // Hillis-Steele in both latency and code complexity.
  if (tid == 0) {
    int sum = 0;
    for (int t = 0; t < kNonZeroBlockSize; ++t) {
      s_chunk_offset[t] = sum;
      sum += s_chunk_count[t];
    }
    *count_ptr = sum;
  }
  __syncthreads();

  // Phase 3: each thread re-walks its chunk and writes coords.
  int local = 0;
  int base = s_chunk_offset[tid];
  for (int64_t i = lo; i < hi; ++i) {
    if (input[i] != (T)0) {
      int pos = base + local;
      ++local;

      // Decompose flat index -> multi-dim coordinates (row-major).
      int64_t remaining = i;
      for (int d = rank - 1; d >= 0; --d) {
        int64_t dim_size = dims.d[d];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        output[d * capacity + pos] = coord;
      }
    }
  }
}

// Host-callable launcher
extern "C" int hip_nonzero(void *stream, const void *input, void *output,
                           int *count_ptr, int64_t input_num_elements,
                           int64_t input_rank,
                           const int64_t *input_dims_host,
                           int64_t output_capacity, int hip_dtype) {
  if (input_num_elements <= 0 || input_rank <= 0 || input_rank > kMaxRank)
    return -1;

  hipStream_t s = static_cast<hipStream_t>(stream);

  // Zero the output buffer so padded slots (beyond actual count) contain
  // index [0,...,0]. Important: ScatterND with valid_count gating only
  // touches the first count slots, but defence-in-depth -- some passes
  // may walk the full capacity (e.g. Transpose) and stale data could
  // surface in indirect reads.
  hipMemsetAsync(output, 0, input_rank * output_capacity * sizeof(int64_t), s);

  // Pack dims into a by-value struct (no device buffer, no per-call hipMalloc).
  NonZeroDims dims;
  for (int d = 0; d < (int)input_rank; ++d)
    dims.d[d] = input_dims_host[d];

  // Single block, kNonZeroBlockSize threads -- cooperative ordered scan.
  // Each thread handles ceil(numel / kNonZeroBlockSize) elements.
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    nonzero_ordered_kernel<__half><<<1, kNonZeroBlockSize, 0, s>>>(
        (const __half *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  case HIP_DTYPE_FLOAT32:
    nonzero_ordered_kernel<float><<<1, kNonZeroBlockSize, 0, s>>>(
        (const float *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  case HIP_DTYPE_INT32:
    nonzero_ordered_kernel<int><<<1, kNonZeroBlockSize, 0, s>>>(
        (const int *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  case HIP_DTYPE_INT64:
    nonzero_ordered_kernel<int64_t><<<1, kNonZeroBlockSize, 0, s>>>(
        (const int64_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  case HIP_DTYPE_INT8:
    nonzero_ordered_kernel<int8_t><<<1, kNonZeroBlockSize, 0, s>>>(
        (const int8_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  case HIP_DTYPE_UINT8:
    nonzero_ordered_kernel<uint8_t><<<1, kNonZeroBlockSize, 0, s>>>(
        (const uint8_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, dims);
    break;
  default:
    return -1;
  }

  return 0;
}
