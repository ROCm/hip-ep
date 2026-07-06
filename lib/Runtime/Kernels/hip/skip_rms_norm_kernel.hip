/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Fused SkipSimplifiedLayerNormalization (RMSNorm) as one custom HIP kernel.
//
//   tmp    = input + skip [+ bias]
//   output = tmp * rsqrt(mean(tmp^2) + eps) * gamma
//   (optional) input_skip_bias_sum = tmp
//
// Replaces the MIOpen composition (miopenOpTensor ADD [+ADD] +
// miopenT5LayerNormForward = 2-3 launches, each high-overhead on gfx1151) with
// a single launch. One block per row; tmp is staged in LDS (fp32) so pass-2
// reuses it without re-reading global. fp32 accumulation matches MIOpen's T5LN
// (RMSNorm) math for numerical parity. f16 only + hidden small enough for LDS;
// caller falls back to MIOpen otherwise.
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <int BS>
__global__ void skip_rms_norm_f16_kernel(
    const __half *__restrict__ input, const __half *__restrict__ skip,
    const __half *__restrict__ gamma, const __half *__restrict__ bias,
    __half *__restrict__ output, __half *__restrict__ skip_sum, int hidden,
    float eps) {
  extern __shared__ float s_tmp[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __half *xin = input + static_cast<size_t>(row) * hidden;
  const __half *sk = skip + static_cast<size_t>(row) * hidden;
  __half *out = output + static_cast<size_t>(row) * hidden;
  __half *ssum = skip_sum ? skip_sum + static_cast<size_t>(row) * hidden : nullptr;

  float ss = 0.0f;
  for (int j = tid; j < hidden; j += BS) {
    float t = __half2float(xin[j]) + __half2float(sk[j]);
    if (bias)
      t += __half2float(bias[j]);
    s_tmp[j] = t;
    ss += t * t;
    if (ssum)
      ssum[j] = __float2half(t);
  }

  // Block reduce sum-of-squares (warp shuffle + LDS across warps).
  constexpr int WARP = 32;
  constexpr int NW = BS / WARP;
  for (int off = WARP >> 1; off > 0; off >>= 1)
    ss += __shfl_down(ss, off);
  __shared__ float warp_ss[NW];
  if ((tid % WARP) == 0)
    warp_ss[tid / WARP] = ss;
  __syncthreads();
  __shared__ float inv_rms;
  if (tid == 0) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; w++)
      tot += warp_ss[w];
    inv_rms = rsqrtf(tot / hidden + eps);
  }
  __syncthreads();

  const float r = inv_rms;
  for (int j = tid; j < hidden; j += BS)
    out[j] = __float2half(s_tmp[j] * r * __half2float(gamma[j]));
}

// Returns -2 if unsupported (caller falls back to MIOpen), 0 ok, else launch err.
extern "C" int hip_skip_rms_norm(void *stream, const void *input,
                                 const void *skip, const void *gamma,
                                 const void *bias, void *output, void *skip_sum,
                                 int64_t num_rows, int64_t hidden,
                                 int64_t element_size_bytes, float eps) {
  if (num_rows <= 0 || hidden <= 0)
    return 0;
  if (element_size_bytes != 2)
    return -2; // f16 only; MIOpen handles f32
  // LDS budget: hidden floats. gfx1151 workgroup LDS = 64 KB -> hidden<=16384.
  if (hidden > 16384)
    return -2;
  constexpr int BS = 256;
  size_t shmem = static_cast<size_t>(hidden) * sizeof(float);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(skip_rms_norm_f16_kernel<BS>),
                     dim3(static_cast<unsigned>(num_rows)), dim3(BS), shmem,
                     static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(input),
                     static_cast<const __half *>(skip),
                     static_cast<const __half *>(gamma),
                     static_cast<const __half *>(bias),
                     static_cast<__half *>(output),
                     static_cast<__half *>(skip_sum), static_cast<int>(hidden),
                     eps);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_skip_rms_norm launch failed: %s\n",
            hipGetErrorString(e));
  return static_cast<int>(e);
}
