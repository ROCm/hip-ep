/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// SkipLayerNormalization (fused HIP kernel)
//===----------------------------------------------------------------------===//
//
// Matches com.microsoft SkipLayerNormalization (ONNX Runtime contrib):
//   sum = input + skip
//   output = LayerNormalization(sum) * gamma + beta
//
// LayerNorm on the last dimension (hidden_dim):
//   mean = mean(sum)
//   var  = mean(sum^2) - mean^2   (population variance; clamped to >= 0)
//   y_j = (sum_j - mean) * rsqrt(var + epsilon) * gamma_j + beta_j
//
// beta is optional (omit by passing NULL; equivalent to zeros).
// Optional input_skip_sum: when non-null, writes fused input+skip (same layout
// as input) for downstream residual use.
//
// Performance: when (hidden_dim + kBlockSize) floats fit in per-block shared
// memory, a "staged" kernel stores fused x = input + skip in shared memory so
// the output pass reads x from shared instead of reloading input/skip from
// global memory. Otherwise the legacy two-pass global-load path is used.
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

/// Legacy path: reads input/skip twice from global (no shared row cache).
template <typename ElemT, int BlockDim>
__global__ void skip_layer_norm_row_kernel(
    const ElemT *__restrict__ input, const ElemT *__restrict__ skip,
    const ElemT *__restrict__ gamma, const ElemT *__restrict__ beta,
    ElemT *__restrict__ output, ElemT *__restrict__ input_skip_sum,
    int64_t hidden_dim, float epsilon, int has_beta, int write_sum) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_off = row * hidden_dim;

  __shared__ float reduce_buf[BlockDim];

  float partial_s = 0.f;
  float partial_s2 = 0.f;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = load_as_float(input + row_off + j) +
              load_as_float(skip + row_off + j);
    if (write_sum)
      store_from_float(input_skip_sum + row_off + j, x);
    partial_s += x;
    partial_s2 += x * x;
  }

  float sum_x = block_reduce_sum<BlockDim>(partial_s, reduce_buf);
  float sum_x2 = block_reduce_sum<BlockDim>(partial_s2, reduce_buf);

  float inv_h = 1.f / static_cast<float>(hidden_dim);
  float mean = sum_x * inv_h;
  float mean_sq = sum_x2 * inv_h;
  float var = mean_sq - mean * mean;
  if (var < 0.f)
    var = 0.f;
  float inv_std = rsqrtf(var + epsilon);

  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = load_as_float(input + row_off + j) +
              load_as_float(skip + row_off + j);
    float g = load_as_float(gamma + j);
    float b = has_beta ? load_as_float(beta + j) : 0.f;
    float y = (x - mean) * inv_std * g + b;
    store_from_float(output + row_off + j, y);
  }
}

/// Staged path: dynamic shared layout [row_cache: hidden_dim floats]
/// [reduce_buf: BlockDim floats]. Fused x is written once; output pass uses
/// shared row_cache only for x (gamma/beta still global).
template <typename ElemT, int BlockDim>
__global__ void skip_layer_norm_row_kernel_staged(
    const ElemT *__restrict__ input, const ElemT *__restrict__ skip,
    const ElemT *__restrict__ gamma, const ElemT *__restrict__ beta,
    ElemT *__restrict__ output, ElemT *__restrict__ input_skip_sum,
    int64_t hidden_dim, float epsilon, int has_beta, int write_sum) {
  extern __shared__ float smem[];
  float *__restrict__ row_cache = smem;
  float *__restrict__ reduce_buf = smem + hidden_dim;

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_off = row * hidden_dim;

  float partial_s = 0.f;
  float partial_s2 = 0.f;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = load_as_float(input + row_off + j) +
              load_as_float(skip + row_off + j);
    row_cache[j] = x;
    if (write_sum)
      store_from_float(input_skip_sum + row_off + j, x);
    partial_s += x;
    partial_s2 += x * x;
  }
  __syncthreads();

  float sum_x = block_reduce_sum<BlockDim>(partial_s, reduce_buf);
  float sum_x2 = block_reduce_sum<BlockDim>(partial_s2, reduce_buf);

  float inv_h = 1.f / static_cast<float>(hidden_dim);
  float mean = sum_x * inv_h;
  float mean_sq = sum_x2 * inv_h;
  float var = mean_sq - mean * mean;
  if (var < 0.f)
    var = 0.f;
  float inv_std = rsqrtf(var + epsilon);

  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden_dim;
       j += static_cast<int64_t>(BlockDim)) {
    float x = row_cache[j];
    float g = load_as_float(gamma + j);
    float b = has_beta ? load_as_float(beta + j) : 0.f;
    float y = (x - mean) * inv_std * g + b;
    store_from_float(output + row_off + j, y);
  }
}

} // namespace

extern "C" int hip_skip_layer_norm(void *stream, const void *input,
                                   const void *skip, const void *gamma,
                                   const void *beta, void *output,
                                   void *input_skip_sum, int64_t num_rows,
                                   int64_t hidden_dim, int hip_dtype,
                                   float epsilon) {
  if (num_rows <= 0 || hidden_dim <= 0)
    return 0;
  if (!input || !skip || !gamma || !output)
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  const int has_beta = beta ? 1 : 0;
  const int write_sum = input_skip_sum ? 1 : 0;

  dim3 grid(static_cast<unsigned>(num_rows));
  dim3 block(kBlockSize);

  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    device = 0;
  int max_shared_int = 0;
  if (hipDeviceGetAttribute(&max_shared_int,
                            hipDeviceAttributeMaxSharedMemoryPerBlock,
                            device) != hipSuccess)
    max_shared_int = 48 * 1024;
  const size_t max_shared =
      max_shared_int > 0 ? static_cast<size_t>(max_shared_int)
                         : static_cast<size_t>(48 * 1024);
  const size_t staging_bytes =
      (static_cast<size_t>(hidden_dim) + static_cast<size_t>(kBlockSize)) *
      sizeof(float);
  const bool use_row_staging = staging_bytes <= max_shared;

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_skip_layer_norm: f32 rows=%lld hidden=%lld "
        "eps=%e beta=%d sum_out=%d grid=%u block=%u staging=%s "
        "(smem=%zu max=%zu)\n",
        (long long)num_rows, (long long)hidden_dim, (double)epsilon, has_beta,
        write_sum, grid.x, block.x, use_row_staging ? "yes" : "no",
        use_row_staging ? staging_bytes : 0u, max_shared);
    if (use_row_staging) {
      hipLaunchKernelGGL(
          (skip_layer_norm_row_kernel_staged<float, kBlockSize>), grid, block,
          staging_bytes, hip_stream, static_cast<const float *>(input),
          static_cast<const float *>(skip), static_cast<const float *>(gamma),
          has_beta ? static_cast<const float *>(beta) : nullptr,
          static_cast<float *>(output),
          write_sum ? static_cast<float *>(input_skip_sum) : nullptr,
          hidden_dim, epsilon, has_beta, write_sum);
    } else {
      hipLaunchKernelGGL(
          (skip_layer_norm_row_kernel<float, kBlockSize>), grid, block, 0,
          hip_stream, static_cast<const float *>(input),
          static_cast<const float *>(skip), static_cast<const float *>(gamma),
          has_beta ? static_cast<const float *>(beta) : nullptr,
          static_cast<float *>(output),
          write_sum ? static_cast<float *>(input_skip_sum) : nullptr,
          hidden_dim, epsilon, has_beta, write_sum);
    }
    break;
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_skip_layer_norm: f16 rows=%lld hidden=%lld "
        "eps=%e beta=%d sum_out=%d grid=%u block=%u staging=%s "
        "(smem=%zu max=%zu)\n",
        (long long)num_rows, (long long)hidden_dim, (double)epsilon, has_beta,
        write_sum, grid.x, block.x, use_row_staging ? "yes" : "no",
        use_row_staging ? staging_bytes : 0u, max_shared);
    if (use_row_staging) {
      hipLaunchKernelGGL(
          (skip_layer_norm_row_kernel_staged<__half, kBlockSize>), grid, block,
          staging_bytes, hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
          has_beta ? static_cast<const __half *>(beta) : nullptr,
          static_cast<__half *>(output),
          write_sum ? static_cast<__half *>(input_skip_sum) : nullptr,
          hidden_dim, epsilon, has_beta, write_sum);
    } else {
      hipLaunchKernelGGL(
          (skip_layer_norm_row_kernel<__half, kBlockSize>), grid, block, 0,
          hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
          has_beta ? static_cast<const __half *>(beta) : nullptr,
          static_cast<__half *>(output),
          write_sum ? static_cast<__half *>(input_skip_sum) : nullptr,
          hidden_dim, epsilon, has_beta, write_sum);
    }
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_skip_layer_norm: unsupported dtype=%d (use "
            "FLOAT32 or FLOAT16)\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_skip_layer_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
