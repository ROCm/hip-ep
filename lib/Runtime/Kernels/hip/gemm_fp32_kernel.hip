/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>

#include "hip_custom_kernels.h"

//===----------------------------------------------------------------------===//
// Tiled fp32 GEMM (row-major) — vendor-BLAS-disabled fallback for wrap_gemm
//===----------------------------------------------------------------------===//
//
// Computes  out[M,N] = alpha * (A[M,K] @ B[K,N]) + beta * bias_broadcast
//
// Used when hipBLASLt is unavailable (e.g. the functional-model build sets
// HIPDNN_EP_DISABLE_VENDOR_BLAS, so wrap_gemm gets a null hipBLASLt handle).
// There is NO hardware WMMA instruction for fp32 inputs on this GPU family
// (WMMA accepts fp16/bf16/int operands only), so this is a classic
// shared-memory-tiled GEMM rather than a WMMA kernel: a 16x16 output tile per
// block, reducing over 16-deep K tiles staged through LDS (one thread per
// output element). Boundary threads pad with 0, so arbitrary M/N/K are correct
// (no multiple-of-16 requirement, unlike the WMMA path).
//
// Layout matches wrap_gemm's untransposed (transA==transB==0) case: A, B and
// out are all row-major ([M,K], [K,N], [M,N]). bias is broadcast per
// (bias_dim0, bias_dim1) where a dim of 1 means "broadcast along that axis"
// (e.g. a per-N bias is [1, N]); pass bias==nullptr for no bias.

namespace {

constexpr int kTile = 16;

__global__ void __launch_bounds__(kTile *kTile) gemm_fp32_tiled_kernel(
    const float *__restrict__ A, const float *__restrict__ B,
    const float *__restrict__ bias, float *__restrict__ out, int M, int N,
    int K, float alpha, float beta, int bias_dim0, int bias_dim1) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  const int ty = threadIdx.y; // row within the output tile
  const int tx = threadIdx.x; // col within the output tile
  const int row = blockIdx.y * kTile + ty; // output row index (m)
  const int col = blockIdx.x * kTile + tx; // output col index (n)

  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kTile) {
    // Stage one 16x16 tile of A and B into LDS, zero-padding out-of-range
    // elements so the inner product is correct for ragged M/N/K.
    const int a_col = k0 + tx;
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    const int b_row = k0 + ty;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk)
      acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();
  }

  if (row < M && col < N) {
    float v = alpha * acc;
    if (bias) {
      const int br = (bias_dim0 == 1) ? 0 : row;
      const int bc = (bias_dim1 == 1) ? 0 : col;
      v += beta * bias[br * bias_dim1 + bc];
    }
    out[row * N + col] = v;
  }
}

} // namespace

extern "C" int hip_gemm_fp32(void *stream, const void *A, const void *B,
                             const void *bias, void *out, int M, int N, int K,
                             float alpha, float beta, int bias_dim0,
                             int bias_dim1) {
  dim3 block(kTile, kTile);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  // Clear any stale sticky error so hipGetLastError() below reflects only this
  // launch (a swallowed upstream error must not be misattributed here).
  (void)hipGetLastError();
  hipLaunchKernelGGL(gemm_fp32_tiled_kernel, grid, block, 0,
                     (hipStream_t)stream, (const float *)A, (const float *)B,
                     (const float *)bias, (float *)out, M, N, K, alpha, beta,
                     bias_dim0, bias_dim1);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
