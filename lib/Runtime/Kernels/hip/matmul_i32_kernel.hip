/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>

// hipBLASLt has no int32 GEMM. An f32 accumulator drops bits once a partial
// sum passes 2^24, so this kernel accumulates the dot product in int64 and
// stores a wrapping int32. A is batch-contiguous [batch, M, K]. B's step
// between batches is b_batch_stride elements: 0 reuses one [K, N] matrix,
// K*N walks a batch-contiguous B.
__global__ void matmul_i32_kernel(const int32_t *__restrict__ a,
                                  const int32_t *__restrict__ b,
                                  int32_t *__restrict__ c, int64_t m, int64_t n,
                                  int64_t k, int64_t batch,
                                  int64_t b_batch_stride) {
  const int64_t total = batch * m * n;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t mn = m * n;
    const int64_t batchIndex = idx / mn;
    const int64_t rem = idx - batchIndex * mn;
    const int64_t row = rem / n;
    const int64_t col = rem - row * n;
    const int32_t *aRow = a + (batchIndex * m + row) * k;
    const int32_t *bCol = b + batchIndex * b_batch_stride + col;
    int64_t acc = 0;
    for (int64_t ki = 0; ki < k; ++ki)
      acc += static_cast<int64_t>(aRow[ki]) *
             static_cast<int64_t>(bCol[ki * n]);
    c[idx] = static_cast<int32_t>(acc);
  }
}

extern "C" int hip_matmul_i32(void *stream, const void *a, const void *b,
                              void *c, int64_t m, int64_t n, int64_t k,
                              int64_t batch, int64_t b_batch_stride) {
  if (m < 0 || n < 0 || k < 0 || batch < 0 || b_batch_stride < 0)
    return -1;
  if (m == 0 || n == 0 || batch == 0)
    return 0;

  constexpr int kBlock = 256;
  const int64_t total = batch * m * n;
  int64_t grid = (total + kBlock - 1) / kBlock;
  if (grid > 65535)
    grid = 65535;

  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  hipLaunchKernelGGL(matmul_i32_kernel, dim3(static_cast<unsigned>(grid)),
                     dim3(kBlock), 0, hipStream, static_cast<const int32_t *>(a),
                     static_cast<const int32_t *>(b), static_cast<int32_t *>(c),
                     m, n, k, batch, b_batch_stride);
  return static_cast<int>(hipGetLastError());
}
