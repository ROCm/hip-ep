/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_custom_kernels.h"

#include <cstdint>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

static constexpr int kWmmaTile = 16;  // RDNA 3 WMMA tile dimension (16x16x16)

//===----------------------------------------------------------------------===//
// WMMA GEMM kernel for small-M fp16 MatMuls (M <= 512)
//===----------------------------------------------------------------------===//
//
// Standard ONNX Gemm computed with RDNA 3+ WMMA instructions:
//
//     output[M,N] = alpha * op(A) * op(B) + beta * C
//
// All matrices row-major, fp16 inputs, fp32 accumulation, fp16 output.
// Currently the kernel body covers transA == 0, alpha == beta == 1, an optional
// row-broadcast bias C ([N] / [1,N]), and transB in {0,1}; the host entry
// hip_gemm_wmma_fp16() validates these and falls back otherwise.
//
// Uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 for hardware-accelerated
// 16x16x16 tile multiply-accumulate on RDNA 3+.
//
// Register-tiled: each wavefront computes a (KX*16) x (KY*16) output tile.
// KX A fragments are each reused KY times; KY B fragments each reused KX times.
// Grid: (ceil(N/(KY*16)), ceil(M/(KX*16)))
//
// WMMA fragment layout (wave32, per AMD GPUOpen / tinygrad / GQA kernel):
//   a_frag (half16): a_frag[k] at lane m = A_tile[m, k]
//                     lane = M-row, element = K-col
//   b_frag (half16): b_frag[k] at lane n = B_tile[k, n]
//                     element = K-row, lane = N-col
//   c_frag (float8):  c_frag[ele] -> row (ele*2 + pair), col (lane)
//                     where lane = threadIdx.x % 16, pair = threadIdx.x / 16
//
// Lanes 0-15 and 16-31 load identical data (required by RDNA 3 WMMA),
// automatically satisfied by indexing with threadIdx.x % 16.

// TransB selects the layout of B:
//   TransB == false : B is [K, N] row-major  (ONNX Gemm transB=0)
//   TransB == true  : B is [N, K] row-major  (ONNX Gemm transB=1, i.e. nn.Linear
//                     weight). For a fixed output column the K elements are then
//                     contiguous, so the B fragment can be vector-loaded too.
template<int KX, int KY, bool TransB>
__global__ void __launch_bounds__(32)
gemm_wmma_fp16_kernel(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    const _Float16* __restrict__ bias,  // optional [N] row-broadcast bias (may be null)
    int M, int K, int N)
{
    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int lane = threadIdx.x % kWmmaTile;
    const int pair = threadIdx.x / kWmmaTile;

    const int m_base = m_tile * KX * kWmmaTile;
    const int n_base = n_tile * KY * kWmmaTile;

    half16 a_frag[KX];
    half16 b_frag[KY];
    float8 c_frag[KY][KX] = {};

    for (int k = 0; k < K; k += kWmmaTile) {
        // A fragment: the 16 K-elements (k..k+15) of a given row are contiguous
        // in row-major A, so load them with a single 32-byte vector load instead
        // of 16 scalar loads. K % 16 == 0 is guaranteed by the dispatch guard, so
        // &A[row*K + k] is 32-byte aligned (relative to A) and [k, k+16) stays
        // in-bounds. A's base 32-byte alignment is checked in the host entry.
        #pragma unroll
        for (int x = 0; x < KX; ++x) {
            int row = m_base + x * kWmmaTile + lane;
            if (row < M) {
                a_frag[x] = *reinterpret_cast<const half16*>(&A[row * K + k]);
            } else {
                #pragma unroll
                for (int ele = 0; ele < kWmmaTile; ++ele)
                    a_frag[x][ele] = (_Float16)0;
            }
        }

        if constexpr (TransB) {
            // B is [N, K] row-major (transB=1). For output column `col` the 16
            // K-elements (k..k+15) live contiguously at &B[col*K + k], so load
            // the whole b_frag with one 32-byte vector load. K % 16 == 0 (dispatch
            // guard) keeps the address 32-byte aligned (relative to B) and
            // [k, k+16) in-bounds; B's base alignment is checked in the host entry.
            #pragma unroll
            for (int y = 0; y < KY; ++y) {
                int col = n_base + y * kWmmaTile + lane;
                if (col < N) {
                    b_frag[y] = *reinterpret_cast<const half16*>(&B[col * K + k]);
                } else {
                    #pragma unroll
                    for (int ele = 0; ele < kWmmaTile; ++ele)
                        b_frag[y][ele] = (_Float16)0;
                }
            }
        } else {
            // B is [K, N] row-major (transB=0): along the K dimension B is strided
            // by N, so it cannot be vectorized, but the per-lane column access is
            // already coalesced across the wave. Keep the scalar element loads.
            #pragma unroll
            for (int ele = 0; ele < kWmmaTile; ++ele) {
                #pragma unroll
                for (int y = 0; y < KY; ++y) {
                    int col = n_base + y * kWmmaTile + lane;
                    b_frag[y][ele] = (col < N) ? B[(k + ele) * N + col]
                                               : (_Float16)0;
                }
            }
        }

        #pragma unroll
        for (int y = 0; y < KY; ++y)
            #pragma unroll
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                   a_frag[x], b_frag[y], c_frag[y][x]);
    }

    #pragma unroll
    for (int y = 0; y < KY; ++y) {
        int c = n_base + y * kWmmaTile + lane;
        // bias is broadcast along M (one value per output column). Load it once
        // per column here; bv is 0 when no bias so the add below is unconditional.
        float bv = (bias && c < N) ? (float)bias[c] : 0.0f;
        #pragma unroll
        for (int x = 0; x < KX; ++x) {
            #pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + ele * 2 + pair;
                if (r < M && c < N)
                    C[r * N + c] = (_Float16)(c_frag[y][x][ele] + bv);
            }
        }
    }
}

// Standard ONNX Gemm entry: output = alpha*op(A)*op(B) + beta*C.
//
// The caller (wrap_gemm) only guarantees fp16 data with K % 16 == 0 and
// N % 16 == 0. Every other ONNX Gemm attribute is validated here; if this
// kernel cannot handle the requested configuration we return a positive code
// and the caller falls back to hipBLASLt. Supported today:
//   * transA == 0
//   * alpha  == 1.0f, beta == 1.0f
//   * C absent (no bias), or C a row-broadcast bias [N] / [1,N]
//     (cDim0 == 1 && cDim1 == N)
//   * transB == 0 (B is [K,N]) or transB == 1 (B is [N,K])
extern "C"
int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                       const void* C, void* output, int M, int N, int K,
                       float alpha, float beta, int transA, int transB,
                       int cDim0, int cDim1) {
    // ---- Capability checks (return >0 => "not handled, please fall back") ----
    if (transA != 0)
        return 1;
    if (alpha != 1.0f || beta != 1.0f)
        return 1;

    // The fragment loads use 32-byte (half16) vector reads, which require the
    // matrix base pointer to be 32-byte aligned (the in-row offset is already a
    // multiple of 32 bytes because K % 16 == 0). A is always vector-loaded; B is
    // vector-loaded only when transB == 1. A non-32-byte-aligned base (e.g. an
    // unaligned constant offset) would fault or read garbage, so fall back.
    if (reinterpret_cast<uintptr_t>(A) % 32 != 0)
        return 1;
    if (transB != 0 && reinterpret_cast<uintptr_t>(B) % 32 != 0)
        return 1;

    // C must be either absent or a row-broadcast bias [N] / [1,N].
    const _Float16* bias = nullptr;
    if (C != nullptr) {
        if (cDim0 == 1 && cDim1 == N) {
            bias = reinterpret_cast<const _Float16*>(C);
        } else {
            return 1; // per-row / full / scalar C not supported here
        }
    }

    // ---- Launch ----
    constexpr int KX = 2;
    constexpr int KY = 4;
    dim3 block(32);
    dim3 grid((N + KY * kWmmaTile - 1) / (KY * kWmmaTile),
              (M + KX * kWmmaTile - 1) / (KX * kWmmaTile));
    if (transB) {
        gemm_wmma_fp16_kernel<KX, KY, true><<<grid, block, 0, (hipStream_t)stream>>>(
            (const _Float16*)A, (const _Float16*)B, (_Float16*)output,
            bias, M, K, N);
    } else {
        gemm_wmma_fp16_kernel<KX, KY, false><<<grid, block, 0, (hipStream_t)stream>>>(
            (const _Float16*)A, (const _Float16*)B, (_Float16*)output,
            bias, M, K, N);
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
