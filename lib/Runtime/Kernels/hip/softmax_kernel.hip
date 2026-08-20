/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- softmax_kernel.hip - Typed row-wise 2D softmax --------------------===//
//
// Standalone Softmax over the last dim of a flattened [rows, cols] f16, bf16,
// or f32 row-major buffer. One block per row, with f32 max/sum accumulation.
// Matches ONNX Softmax semantics for the final dimension.
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

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Pulls in the HIP_KERNEL_API (dllexport) declaration of
// hip_softmax_row_2d_inplace so this TU exports it from the per-arch SHARED
// kernel DLL (the native artifact link resolves it via the import lib).
#include "hip_custom_kernels.h"

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

template <typename T> struct SoftmaxIO;

template <> struct SoftmaxIO<float> {
  __device__ static inline float load(float value) { return value; }
  __device__ static inline float store(float value) { return value; }
};

template <> struct SoftmaxIO<__half> {
  __device__ static inline float load(__half value) {
    return __half2float(value);
  }
  __device__ static inline __half store(float value) {
    return __float2half(value);
  }
};

template <> struct SoftmaxIO<__hip_bfloat16> {
  __device__ static inline float load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static inline __hip_bfloat16 store(float value) {
    return __float2bfloat16(value);
  }
};

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

template <typename T>
__global__ void softmax_row_2d_inplace_kernel(T* data, int rows,
                                               int cols) {
  const int row = blockIdx.x;
  if (row >= rows)
    return;
  T* row_ptr = data + (size_t)row * cols;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: row max.
  float local_max = -INFINITY;
  for (int j = tid; j < cols; j += blockDim.x)
    local_max = fmaxf(local_max, SoftmaxIO<T>::load(row_ptr[j]));
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
    float e = exp2f((SoftmaxIO<T>::load(row_ptr[j]) - row_max) * LOG2E);
    row_ptr[j] = SoftmaxIO<T>::store(e);
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
    row_ptr[j] = SoftmaxIO<T>::store(SoftmaxIO<T>::load(row_ptr[j]) * inv);
}

template <typename T>
int launchSoftmaxRow2D(void* stream_ptr, void* data, int rows, int cols) {
  if (!data || rows <= 0 || cols <= 0)
    return -1;

  // A block must contain at least one complete hardware wave. Starting at
  // WAVE_SIZE also fixes small-column launches on wave64 targets.
  int threads = WAVE_SIZE;
  while (threads < cols && threads < 256)
    threads *= 2;
  int num_waves = threads / WAVE_SIZE;
  size_t smem_bytes = static_cast<size_t>(num_waves) * sizeof(float);
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_row_2d_inplace_kernel<T>),
                     dim3(rows), dim3(threads), smem_bytes, stream,
                     static_cast<T*>(data), rows, cols);
  return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_softmax_row_2d_inplace(void* stream, void* data, int rows,
                                           int cols) {
  return launchSoftmaxRow2D<__half>(stream, data, rows, cols);
}

extern "C" int hip_softmax_row_2d_inplace_fp32(void* stream, void* data,
                                                int rows, int cols) {
  return launchSoftmaxRow2D<float>(stream, data, rows, cols);
}

extern "C" int hip_softmax_row_2d_inplace_bf16(void* stream, void* data,
                                                int rows, int cols) {
  return launchSoftmaxRow2D<__hip_bfloat16>(stream, data, rows, cols);
}
