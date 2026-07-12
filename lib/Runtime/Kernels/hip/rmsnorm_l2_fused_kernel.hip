/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Fused ORCA RMS-norm (L2 form): out[r,i] = weight[i] * x[r,i] / sqrt(sum_i x[r,i]^2)
 *
 * This is the exact fused form of the decomposed chain the ORCA decode graph
 * emits (mul(x,x) -> reduce_sum -> sqrt -> reciprocal -> mul(x,.) -> mul(w,.)).
 * The RMSNorm 1/N mean factor and any epsilon were folded into `weight` at
 * export, so this kernel deliberately uses sum (not mean) and no epsilon --
 * matching the decomposed ops bit-for-bit up to reduction order (all fp32).
 *
 * Layout: input reshaped to [outer, norm_size] on the host (norm_size = H =
 * weight_num_elements). One block per row, shared-memory reduction, fp32 math.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kRmsL2BlockSize = 256;

// out[r,i] = weight[i] * x[r,i] * rsqrt(sum_i x[r,i]^2)   (fp32 in/out)
__global__ void rmsnorm_l2_fused_kernel(const float *__restrict__ x,
                                        const float *__restrict__ weight,
                                        float *__restrict__ out,
                                        int64_t norm_size) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float *xr = x + row * norm_size;
  float *outr = out + row * norm_size;

  // Pass A: sum of squares in fp32.
  float local = 0.0f;
  for (int64_t i = tid; i < norm_size; i += kRmsL2BlockSize) {
    float v = xr[i];
    local += v * v;
  }
  __shared__ float smem[kRmsL2BlockSize];
  smem[tid] = local;
  __syncthreads();
  for (int s = kRmsL2BlockSize >> 1; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  // rsqrt(sum(x^2)); guard the degenerate all-zero row.
  __shared__ float rnorm;
  if (tid == 0) {
    float ss = smem[0];
    rnorm = ss > 0.0f ? (1.0f / sqrtf(ss)) : 0.0f;
  }
  __syncthreads();

  // Pass B: scale by weight and rnorm.
  const float r = rnorm;
  for (int64_t i = tid; i < norm_size; i += kRmsL2BlockSize) {
    outr[i] = weight[i] * xr[i] * r;
  }
}

// Launcher used by the runtime wrap (see real/rmsnorm_l2_fused.cpp).
extern "C" int hip_rmsnorm_l2_fused(void *stream, const void *x,
                                    const void *weight, void *out,
                                    int64_t outer, int64_t norm_size) {
  if (!x || !weight || !out || outer <= 0 || norm_size <= 0)
    return -1;
  dim3 grid(static_cast<unsigned>(outer));
  dim3 block(kRmsL2BlockSize);
  (void)hipGetLastError();
  hipLaunchKernelGGL(rmsnorm_l2_fused_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const float *>(x),
                     static_cast<const float *>(weight),
                     static_cast<float *>(out), norm_size);
  return static_cast<int>(hipGetLastError());
}
