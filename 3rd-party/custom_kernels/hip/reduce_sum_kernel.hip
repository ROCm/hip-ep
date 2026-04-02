/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Parallel sum reduction for i64. Each block reduces one output element
// by summing reduce_size consecutive input elements using shared memory.
__global__ void reduce_sum_i64_kernel0(
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

// Parallel sum reduction for i64. Each block reduces one output element
// by summing reduce_size consecutive input elements using warp shuffles and
// a small shared-memory staging area for cross-warp reduction.
__device__ inline int64_t warp_reduce_sum_i64(int64_t value) {
  const unsigned long long mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__global__ void reduce_sum_i64_kernel1(
    const int64_t* __restrict__ data,
    int64_t* __restrict__ output,
    int64_t reduce_size,
    int64_t num_output_elements) {
  extern __shared__ int64_t sdata[];

  int64_t out_idx = blockIdx.x;
  if (out_idx >= num_output_elements) return;

  const int64_t* slice = data + out_idx * reduce_size;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % warpSize;
  const int warp_id = tid / warpSize;
  const int num_warps = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  const int64_t stride = static_cast<int64_t>(blockDim.x);

  // Each thread accumulates a partial sum over its strided portion. When the
  // slice is 16-byte aligned we read two i64 values at a time, otherwise we
  // fall back to an unrolled scalar loop.
  int64_t sum = 0;
  const bool use_vectorized =
      (reinterpret_cast<std::uintptr_t>(slice) & (alignof(longlong2) - 1)) == 0;

  if (use_vectorized) {
    const int64_t vec_size = reduce_size / 2;
    const int64_t vec_stride = stride;
    const longlong2* slice_vec = reinterpret_cast<const longlong2*>(slice);

    int64_t vec_idx = tid;
    for (; vec_idx + vec_stride < vec_size; vec_idx += vec_stride * 2) {
      const longlong2 v0 = slice_vec[vec_idx];
      const longlong2 v1 = slice_vec[vec_idx + vec_stride];
      sum += static_cast<int64_t>(v0.x) + static_cast<int64_t>(v0.y);
      sum += static_cast<int64_t>(v1.x) + static_cast<int64_t>(v1.y);
    }
    for (; vec_idx < vec_size; vec_idx += vec_stride) {
      const longlong2 v = slice_vec[vec_idx];
      sum += static_cast<int64_t>(v.x) + static_cast<int64_t>(v.y);
    }

    if (((reduce_size & 1) != 0) && tid == 0) {
      sum += slice[reduce_size - 1];
    }
  } else {
    int64_t i = tid;
    for (; i + stride < reduce_size; i += stride * 2) {
      sum += slice[i];
      sum += slice[i + stride];
    }
    for (; i < reduce_size; i += stride) {
      sum += slice[i];
    }
  }

  sum = warp_reduce_sum_i64(sum);

  if (lane == 0) {
    sdata[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    int64_t block_sum = lane < num_warps ? sdata[lane] : 0;
    block_sum = warp_reduce_sum_i64(block_sum);

    if (tid == 0) {
      output[out_idx] = block_sum;
    }
  }
}

extern "C" int hip_reduce_sum(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype) {
  if (num_output_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  if (num_input_elements % num_output_elements != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_reduce_sum: input(%lld) not divisible "
            "by output(%lld)\n",
            (long long)num_input_elements, (long long)num_output_elements);
    return -1;
  }

  int64_t reduce_size = num_input_elements / num_output_elements;

  switch (hip_dtype) {
    case HIP_DTYPE_INT64: {
      int block_size = 256;
      if (reduce_size < 256) {
        block_size = 1;
        while (block_size < reduce_size) block_size <<= 1;
        if (block_size < 1) block_size = 1;
      }
      size_t shared_mem = block_size * sizeof(int64_t);

      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_reduce_sum: dtype=INT64, "
              "input=%lld, output=%lld, reduce_size=%lld, "
              "grid=%lld, block=%d, smem=%zu\n",
              (long long)num_input_elements, (long long)num_output_elements,
              (long long)reduce_size, (long long)num_output_elements,
              block_size, shared_mem);

      if(reduce_size >= 2048){
        hipLaunchKernelGGL(reduce_sum_i64_kernel1,
                         dim3(static_cast<unsigned>(num_output_elements)),
                         dim3(block_size),
                         shared_mem, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<int64_t*>(output),
                         reduce_size, num_output_elements);

      }
      else{
        hipLaunchKernelGGL(reduce_sum_i64_kernel0,
                         dim3(static_cast<unsigned>(num_output_elements)),
                         dim3(block_size),
                         shared_mem, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<int64_t*>(output),
                         reduce_size, num_output_elements);

      }
      
      break;
    }
    default:
      fprintf(stderr,
              "[custom_kernels] hip_reduce_sum: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_reduce_sum launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
