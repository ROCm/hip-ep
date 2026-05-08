/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// INT64 reductions
// =============================================================================

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

__global__ void reduce_sum_i64_shfl_kernel(
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

// =============================================================================
// INT32 reduction
// =============================================================================

// Parallel sum reduction for i32. Accumulates in int64 to avoid overflow when
// reducing large slices, then narrows to int32 for the final write. Matches
// ONNX ReduceSum semantics where the output type equals the input type but
// intermediate accumulation should not silently wrap.
__global__ void reduce_sum_i32_kernel(
    const int32_t* __restrict__ data,
    int32_t* __restrict__ output,
    int64_t reduce_size,
    int64_t num_output_elements) {
  extern __shared__ int64_t sdata_i32[];

  int64_t out_idx = blockIdx.x;
  if (out_idx >= num_output_elements) return;

  const int32_t* slice = data + out_idx * reduce_size;
  int tid = threadIdx.x;

  int64_t sum = 0;
  for (int64_t i = tid; i < reduce_size; i += blockDim.x) {
    sum += static_cast<int64_t>(slice[i]);
  }
  sdata_i32[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      sdata_i32[tid] += sdata_i32[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[out_idx] = static_cast<int32_t>(sdata_i32[0]);
  }
}

// =============================================================================
// FP16 reduction
// =============================================================================

// Parallel sum reduction for fp16. Accumulates in float to avoid the precision
// loss / overflow that would occur with a half accumulator over long slices,
// then narrows back to half for the final write. This mirrors how cuDNN/MIOpen
// implement fp16 reductions.
__global__ void reduce_sum_f16_kernel(
    const __half* __restrict__ data,
    __half* __restrict__ output,
    int64_t reduce_size,
    int64_t num_output_elements) {
  extern __shared__ float sdata_f16[];

  int64_t out_idx = blockIdx.x;
  if (out_idx >= num_output_elements) return;

  const __half* slice = data + out_idx * reduce_size;
  int tid = threadIdx.x;

  float sum = 0.0f;
  for (int64_t i = tid; i < reduce_size; i += blockDim.x) {
    sum += __half2float(slice[i]);
  }
  sdata_f16[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      sdata_f16[tid] += sdata_f16[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[out_idx] = __float2half(sdata_f16[0]);
  }
}

// =============================================================================
// Host launcher
// =============================================================================

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

  // Block size derivation shared by all dtype branches: cap at 256, then round
  // down to the next power of two when reduce_size is small so the in-kernel
  // tree reduction has a power-of-two range.
  auto pick_block_size = [reduce_size]() {
    int block_size = 256;
    if (reduce_size < 256) {
      block_size = 1;
      while (block_size < reduce_size) block_size <<= 1;
      if (block_size < 1) block_size = 1;
    }
    return block_size;
  };

  switch (hip_dtype) {
    case HIP_DTYPE_INT64: {
      int block_size = pick_block_size();
      size_t shared_mem = block_size * sizeof(int64_t);

      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_reduce_sum: dtype=INT64, "
              "input=%lld, output=%lld, reduce_size=%lld, "
              "grid=%lld, block=%d, smem=%zu\n",
              (long long)num_input_elements, (long long)num_output_elements,
              (long long)reduce_size, (long long)num_output_elements,
              block_size, shared_mem);

      if(reduce_size >= 2048){
        hipLaunchKernelGGL(reduce_sum_i64_shfl_kernel,
                         dim3(static_cast<unsigned>(num_output_elements)),
                         dim3(block_size),
                         shared_mem, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<int64_t*>(output),
                         reduce_size, num_output_elements);
      }
      else{
        hipLaunchKernelGGL(reduce_sum_i64_kernel,
                         dim3(static_cast<unsigned>(num_output_elements)),
                         dim3(block_size),
                         shared_mem, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<int64_t*>(output),
                         reduce_size, num_output_elements);
      }
      break;
    }
    case HIP_DTYPE_INT32: {
      int block_size = pick_block_size();
      // Shared mem holds int64 accumulators (one per thread) for overflow-safe
      // accumulation; output is narrowed to int32 by the kernel.
      size_t shared_mem = block_size * sizeof(int64_t);

      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_reduce_sum: dtype=INT32, "
              "input=%lld, output=%lld, reduce_size=%lld, "
              "grid=%lld, block=%d, smem=%zu\n",
              (long long)num_input_elements, (long long)num_output_elements,
              (long long)reduce_size, (long long)num_output_elements,
              block_size, shared_mem);

      hipLaunchKernelGGL(reduce_sum_i32_kernel,
                       dim3(static_cast<unsigned>(num_output_elements)),
                       dim3(block_size),
                       shared_mem, hip_stream,
                       static_cast<const int32_t*>(data),
                       static_cast<int32_t*>(output),
                       reduce_size, num_output_elements);
      break;
    }
    case HIP_DTYPE_FLOAT16: {
      int block_size = pick_block_size();
      // Shared mem holds float accumulators (one per thread) to avoid fp16
      // precision loss; output is narrowed to half by the kernel.
      size_t shared_mem = block_size * sizeof(float);

      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_reduce_sum: dtype=FLOAT16, "
              "input=%lld, output=%lld, reduce_size=%lld, "
              "grid=%lld, block=%d, smem=%zu\n",
              (long long)num_input_elements, (long long)num_output_elements,
              (long long)reduce_size, (long long)num_output_elements,
              block_size, shared_mem);

      hipLaunchKernelGGL(reduce_sum_f16_kernel,
                       dim3(static_cast<unsigned>(num_output_elements)),
                       dim3(block_size),
                       shared_mem, hip_stream,
                       static_cast<const __half*>(data),
                       static_cast<__half*>(output),
                       reduce_size, num_output_elements);
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
