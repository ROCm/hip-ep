/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// SkipSimplifiedLayerNormalization (fused HIP kernel)
//===----------------------------------------------------------------------===//
//
// ONNX com.microsoft SkipSimplifiedLayerNormalization:
//   sum = input + skip [+ bias]
//   output = SimplifiedLayerNormalization(sum) * gamma
//
// SimplifiedLayerNormalization (RMS norm, no mean centering):
//   inv_std = rsqrt(mean(sum^2) + epsilon)   with mean over the last dim
//   output_j = sum_j * inv_std * gamma_j
//
// Optional input_skip_bias_sum: when non-null, writes "sum" (same layout/dtype
// as input) so callers can fuse without a separate add kernel.
//
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float load_as_float(const float *p) {
  return *p;
}
__device__ __forceinline__ float load_as_float(const __half *p) {
  return __half2float(*p);
}
__device__ __forceinline__ void store_from_float(float *p, float v) { *p = v; }
__device__ __forceinline__ void store_from_float(__half *p, float v) {
  *p = __float2half(v);
}

// Block-wide reduction: each thread holds one partial sum; result in thread 0's
// register (all threads must call; thread 0 gets total).
template <int BlockDim>
__device__ float block_reduce_sum(float val, float *shared) {
  const unsigned tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();
#pragma unroll
  for (unsigned s = BlockDim / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }
  return shared[0];
}

template <typename ElemT, int BlockDim>
__global__ void skip_simplified_layer_norm_row_kernel(
    const ElemT *__restrict__ input, const ElemT *__restrict__ skip,
    const ElemT *__restrict__ gamma, const ElemT *__restrict__ bias,
    ElemT *__restrict__ output, ElemT *__restrict__ input_skip_bias_sum,
    int64_t hidden_dim, float epsilon, int has_bias, int write_sum) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_off = row * hidden_dim;

  __shared__ float reduce_buf[BlockDim];

  float partial_sq = 0.f;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = load_as_float(input + row_off + j) +
              load_as_float(skip + row_off + j);
    if (has_bias)
      x += load_as_float(bias + j);
    if (write_sum)
      store_from_float(input_skip_bias_sum + row_off + j, x);
    partial_sq += x * x;
  }

  float sum_sq = block_reduce_sum<BlockDim>(partial_sq, reduce_buf);
  float mean_sq = sum_sq / static_cast<float>(hidden_dim);
  float inv_std = rsqrtf(mean_sq + epsilon);

  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = load_as_float(input + row_off + j) +
              load_as_float(skip + row_off + j);
    if (has_bias)
      x += load_as_float(bias + j);
    float g = load_as_float(gamma + j);
    float y = x * inv_std * g;
    store_from_float(output + row_off + j, y);
  }
}

} // namespace

extern "C" int hip_skip_simplified_layer_norm(
    void *stream, const void *input, const void *skip, const void *gamma,
    const void *bias, void *output, void *input_skip_bias_sum, int64_t num_rows,
    int64_t hidden_dim, int hip_dtype, float epsilon) {
  if (num_rows <= 0 || hidden_dim <= 0)
    return 0;
  if (!input || !skip || !gamma || !output)
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  const int has_bias = bias ? 1 : 0;
  const int write_sum = input_skip_bias_sum ? 1 : 0;

  dim3 grid(static_cast<unsigned>(num_rows));
  dim3 block(kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_skip_simplified_layer_norm: f32 rows=%lld "
        "hidden=%lld eps=%e bias=%d sum_out=%d grid=%u block=%u\n",
        (long long)num_rows, (long long)hidden_dim, (double)epsilon, has_bias,
        write_sum, grid.x, block.x);
    hipLaunchKernelGGL(
        skip_simplified_layer_norm_row_kernel<float, kBlockSize>, grid, block,
        0, hip_stream, static_cast<const float *>(input),
        static_cast<const float *>(skip), static_cast<const float *>(gamma),
        has_bias ? static_cast<const float *>(bias) : nullptr,
        static_cast<float *>(output),
        write_sum ? static_cast<float *>(input_skip_bias_sum) : nullptr,
        hidden_dim, epsilon, has_bias, write_sum);
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_skip_simplified_layer_norm: f16 rows=%lld "
        "hidden=%lld eps=%e bias=%d sum_out=%d grid=%u block=%u\n",
        (long long)num_rows, (long long)hidden_dim, (double)epsilon, has_bias,
        write_sum, grid.x, block.x);
    hipLaunchKernelGGL(
        skip_simplified_layer_norm_row_kernel<__half, kBlockSize>, grid,
        block, 0, hip_stream, static_cast<const __half *>(input),
        static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
        has_bias ? static_cast<const __half *>(bias) : nullptr,
        static_cast<__half *>(output),
        write_sum ? static_cast<__half *>(input_skip_bias_sum) : nullptr,
        hidden_dim, epsilon, has_bias, write_sum);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_skip_simplified_layer_norm: unsupported "
            "dtype=%d (use FLOAT32 or FLOAT16)\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_skip_simplified_layer_norm launch failed: "
            "%s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
