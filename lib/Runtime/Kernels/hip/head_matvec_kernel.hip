// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "head_matvec.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {
constexpr int kColumns = 64, kVector = 4, kRows = 4, kSplits = 4;
constexpr int kN = 131072, kK = 2688;

__global__ void head_partial(const half *x, const half *w, float *scratch) {
  int lane = threadIdx.x % kColumns, row = threadIdx.x / kColumns;
  int col = (blockIdx.x * kColumns + lane) * kVector;
  float accum[kVector] = {};
  for (int k = blockIdx.y * kRows + row; k < kK; k += kSplits * kRows) {
    float xv = __half2float(x[k]);
    const half2 *weight = reinterpret_cast<const half2 *>(w + size_t(k) * kN + col);
#pragma unroll
    for (int v = 0; v < kVector / 2; ++v) {
      float2 pair = __half22float2(weight[v]);
      accum[2*v] = __fmaf_rn(xv, pair.x, accum[2*v]);
      accum[2*v+1] = __fmaf_rn(xv, pair.y, accum[2*v+1]);
    }
  }
  __shared__ float values[kColumns * kRows * kVector];
#pragma unroll
  for (int v = 0; v < kVector; ++v) values[threadIdx.x * kVector + v] = accum[v];
  __syncthreads();
  if (row == 0) {
#pragma unroll
    for (int v = 0; v < kVector; ++v) {
      float sum = 0;
#pragma unroll
      for (int r = 0; r < kRows; ++r)
        sum += values[(r * kColumns + lane) * kVector + v];
      scratch[blockIdx.y * kN + col+v] = sum;
    }
  }
}
__global__ void head_finish(const float *scratch, half *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
#pragma unroll
  for (int split = 0; split < kSplits; ++split) sum += scratch[split * kN + col];
  output[col] = __float2half(sum);
}
} // namespace

extern "C" HIP_KERNEL_API int hip_head_matvec_f16(
    void *stream, const void *input, const void *weights, void *output,
    void *scratch, int n, int k) {
  if (!input || !weights || !output || !scratch || n != kN || k != kK ||
      output == input || output == weights || output == scratch ||
      scratch == input || scratch == weights)
    return hipErrorInvalidValue;
  auto s = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(head_partial, dim3(kN/(kColumns*kVector), kSplits),
                    dim3(kColumns*kRows), 0, s, static_cast<const half *>(input),
                    static_cast<const half *>(weights), static_cast<float *>(scratch));
  auto error = hipGetLastError();
  if (error != hipSuccess) return error;
  hipLaunchKernelGGL(head_finish, dim3(kN/256), dim3(256), 0, s,
                    static_cast<const float *>(scratch), static_cast<half *>(output));
  return hipGetLastError();
}
