/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * NonZero kernel: single-pass atomic approach.
 *
 * Each thread checks one input element. If nonzero, atomicAdd on the
 * count_ptr gives the write position. Coordinates are decomposed from
 * the flat index and written into the output buffer at stride = capacity.
 *
 * Output layout: [rank, capacity] of i64. Valid columns are [0, *count_ptr).
 * The output is NOT ordered (positions depend on atomic scheduling), but
 * downstream ScatterND doesn't require ordering.
 *
 * Bounded to input rank <= 8.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "hip_custom_kernels.h"

static constexpr int kMaxRank = 8;
static constexpr int kBlockSize = 256;

template <typename T>
__global__ void nonzero_kernel(const T *__restrict__ input,
                               int64_t *__restrict__ output,
                               int *__restrict__ count_ptr,
                               int64_t capacity, int rank,
                               int64_t numel,
                               const int64_t *__restrict__ dims) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;

  T val = input[idx];
  bool is_nonzero = (val != (T)0);

  if (!is_nonzero)
    return;

  int pos = atomicAdd(count_ptr, 1);

  // Decompose flat index -> multi-dim coordinates (row-major).
  // dims[0..rank-1] are the input shape dimensions.
  int64_t remaining = idx;
  for (int d = rank - 1; d >= 0; --d) {
    int64_t dim_size = dims[d];
    int64_t coord = remaining % dim_size;
    remaining /= dim_size;
    output[d * capacity + pos] = coord;
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

  // Zero the count before launch
  hipMemsetAsync(count_ptr, 0, sizeof(int), s);

  // Zero the output buffer so padded slots (beyond actual count) contain
  // index [0,...,0]. Without this, ScatterND processing all capacity rows
  // (when count_buf optimization is inactive) would read uninitialized indices.
  hipMemsetAsync(output, 0, input_rank * output_capacity * sizeof(int64_t), s);

  // Copy dims to device (small fixed-size buffer, async)
  int64_t *d_dims = nullptr;
  hipMalloc(&d_dims, input_rank * sizeof(int64_t));
  hipMemcpyAsync(d_dims, input_dims_host, input_rank * sizeof(int64_t),
                 hipMemcpyHostToDevice, s);

  int grid = (int)((input_num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    nonzero_kernel<__half><<<grid, kBlockSize, 0, s>>>(
        (const __half *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  case HIP_DTYPE_FLOAT32:
    nonzero_kernel<float><<<grid, kBlockSize, 0, s>>>(
        (const float *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  case HIP_DTYPE_INT32:
    nonzero_kernel<int><<<grid, kBlockSize, 0, s>>>(
        (const int *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  case HIP_DTYPE_INT64:
    nonzero_kernel<int64_t><<<grid, kBlockSize, 0, s>>>(
        (const int64_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  case HIP_DTYPE_INT8:
    nonzero_kernel<int8_t><<<grid, kBlockSize, 0, s>>>(
        (const int8_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  case HIP_DTYPE_UINT8:
    nonzero_kernel<uint8_t><<<grid, kBlockSize, 0, s>>>(
        (const uint8_t *)input, (int64_t *)output, count_ptr, output_capacity,
        (int)input_rank, input_num_elements, d_dims);
    break;
  default:
    hipFree(d_dims);
    return -1;
  }

  // Free device dims after kernel completes (enqueue on stream)
  hipFreeAsync(d_dims, s);
  return 0;
}
