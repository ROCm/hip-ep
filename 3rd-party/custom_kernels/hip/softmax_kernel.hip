/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- softmax_kernel.hip - Row-wise 2D softmax (fp16) --------------------===//
//
// Standalone Softmax over the last dim of a flattened [rows, cols] fp16
// row-major buffer. One block per row, cooperative warp-shuffle reductions
// for max + sum, then an inverse-multiply pass. Matches ONNX Softmax
// semantics (`axis = -1` along the last dim).
//
// Used by the `hip_miopen_softmax` runtime entry point (lib/Runtime/real/
// activation.cpp) for `onnx.Softmax` outside fused attention paths — most
// notably the SigLIP / ViT self-attention scores in vision encoders. The
// pre-existing `softmax_inplace_kernel` in `gqa_kernel.hip` is column-wise
// over an internal layout convention specific to the GQA pipeline; using
// it for the row-wise standalone case requires the caller to launch one
// block per (head, col) pair which the existing `hip_gqa_softmax_inplace`
// launcher does NOT do — hence this distinct kernel.
//
// Numerics: exp2f(x * LOG2E) instead of expf(x) for the native HW
// instruction, fp32 accumulators, single max + single sum sweep per row.
// LDS use: `num_waves * sizeof(float)` per block.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef WAVE_SIZE
#if defined(__GFX9__) || defined(__GFX9_NEXT__)
#define WAVE_SIZE 64
#else
#define WAVE_SIZE 32
#endif
#endif

#ifndef LOG2E
#define LOG2E 1.44269504088896340736f
#endif

namespace {

__device__ inline float warp_reduce_max_f(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v = fmaxf(v, __shfl_xor(v, off));
  return v;
}

__device__ inline float warp_reduce_sum_f(float v) {
  for (int off = WAVE_SIZE / 2; off > 0; off /= 2)
    v += __shfl_xor(v, off);
  return v;
}

__global__ void softmax_row_2d_inplace_kernel(__half* data, int rows,
                                                int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  __half* row_ptr = data + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += blockDim.x)
    local_max = fmaxf(local_max, __half2float(row_ptr[j]));
  local_max = warp_reduce_max_f(local_max);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_max;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : -INFINITY;
    v = warp_reduce_max_f(v);
    smem[0] = v;
  }
  __syncthreads();
  float row_max = smem[0];

  // Pass 2: exp2f + sum.
  float local_sum = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float e = exp2f((__half2float(row_ptr[j]) - row_max) * LOG2E);
    row_ptr[j] = __float2half(e);
    local_sum += e;
  }
  local_sum = warp_reduce_sum_f(local_sum);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_sum;
  __syncthreads();
  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : 0.0f;
    v = warp_reduce_sum_f(v);
    smem[0] = v;
  }
  __syncthreads();
  float row_sum = smem[0];
  float inv = 1.0f / row_sum;

  // Pass 3: normalize.
  for (int j = tid; j < cols; j += blockDim.x)
    row_ptr[j] = __float2half(__half2float(row_ptr[j]) * inv);
}

} // namespace

extern "C" int hip_softmax_row_2d_inplace(void* stream_ptr, void* data,
                                            int rows, int cols) {
  if (!data || rows <= 0 || cols <= 0)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  // Pick block size: power of two up to min(cols, 256). 32 minimum for one
  // wave on RDNA / 64 on CDNA. SoftMax for vision attention has cols up to
  // ~4096, well-served by 256 threads × 16 iters.
  int threads = 32;
  while (threads < cols && threads < 256)
    threads *= 2;
  if (threads > 256)
    threads = 256;
  int num_waves = (threads + WAVE_SIZE - 1) / WAVE_SIZE;
  size_t smem_bytes = static_cast<size_t>(num_waves) * sizeof(float);
  softmax_row_2d_inplace_kernel<<<rows, threads, smem_bytes, stream>>>(
      static_cast<__half*>(data), rows, cols);
  return hipSuccess;
}
