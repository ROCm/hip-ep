// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "small_matvec.h"
#include <hip/hip_runtime.h>

namespace {
// Prefetch loads, but keep one sequential FMA accumulator per output.
// This matches the pinned BLAS router without a split-sum reduction.
__global__ void sequential_router(const float *input, const float *weights,
                                  float *output) {
  constexpr int columns = 4, chunk = 64, n = 128, k = 2688;
  if (threadIdx.x >= columns) return;
  const int col = blockIdx.x * columns + threadIdx.x;
  float sum = 0;
  for (int j = 0; j < k; j += chunk) {
    float x[chunk], w[chunk];
#pragma unroll
    for (int p = 0; p < chunk; ++p) {
      x[p] = input[j+p];
      w[p] = weights[(j+p)*n+col];
    }
#pragma unroll
    for (int p = 0; p < chunk; ++p) sum = __fmaf_rn(x[p], w[p], sum);
  }
  output[col] = sum;
}
} // namespace

extern "C" HIP_KERNEL_API int hip_sequential_router_f32(
    void *stream, const void *input, const void *weights, void *output,
    int n, int k) {
  if (!input || !weights || !output || n != 128 || k != 2688 ||
      output == input || output == weights)
    return hipErrorInvalidValue;
  hipLaunchKernelGGL(sequential_router, dim3(32), dim3(32), 0,
                    static_cast<hipStream_t>(stream), static_cast<const float *>(input),
                    static_cast<const float *>(weights), static_cast<float *>(output));
  return hipGetLastError();
}
