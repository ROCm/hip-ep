/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Parallel sum reduction for i64. Each block reduces one output element
// by summing reduce_size consecutive input elements using shared memory.
__global__ void reduce_sum_i64_kernel(
    const int64_t* __restrict__ data,
    int64_t* __restrict__ output,
    int64_t reduce_size,
    int64_t num_output_elements) {
  extern __shared__ int64_t sdata[];

  int64_t out_idx = blockIdx.x;
  if (out_idx >= num_output_elements) return;

  const int64_t* slice = data + out_idx * reduce_size;
  int tid = threadIdx.x;

  // Each thread accumulates a partial sum over its strided portion
  int64_t sum = 0;
  for (int64_t i = tid; i < reduce_size; i += blockDim.x) {
    sum += slice[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[out_idx] = sdata[0];
  }
}

extern "C" int udna_reduce_sum(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int udna_dtype) {
  if (num_output_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  if (num_input_elements % num_output_elements != 0) {
    fprintf(stderr,
            "[custom_kernels] udna_reduce_sum: input(%lld) not divisible "
            "by output(%lld)\n",
            (long long)num_input_elements, (long long)num_output_elements);
    return -1;
  }

  int64_t reduce_size = num_input_elements / num_output_elements;

  switch (udna_dtype) {
    case UDNA_DTYPE_INT64: {
      int block_size = 256;
      if (reduce_size < 256) {
        block_size = 1;
        while (block_size < reduce_size) block_size <<= 1;
        if (block_size < 1) block_size = 1;
      }
      size_t shared_mem = block_size * sizeof(int64_t);

      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] udna_reduce_sum: dtype=INT64, "
              "input=%lld, output=%lld, reduce_size=%lld, "
              "grid=%lld, block=%d, smem=%zu\n",
              (long long)num_input_elements, (long long)num_output_elements,
              (long long)reduce_size, (long long)num_output_elements,
              block_size, shared_mem);

      hipLaunchKernelGGL(reduce_sum_i64_kernel,
                         dim3(static_cast<unsigned>(num_output_elements)),
                         dim3(block_size),
                         shared_mem, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<int64_t*>(output),
                         reduce_size, num_output_elements);
      break;
    }
    default:
      fprintf(stderr,
              "[custom_kernels] udna_reduce_sum: unsupported dtype=%d\n",
              udna_dtype);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] udna_reduce_sum launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
