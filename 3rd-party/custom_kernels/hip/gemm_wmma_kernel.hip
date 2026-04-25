/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_custom_kernels.h"
#include "wmma_frag.h"

static constexpr int kWmmaTile = WMMA_TILE_K;

//===----------------------------------------------------------------------===//
// WMMA GEMM kernel for small-M fp16 MatMuls (M <= 512)
//===----------------------------------------------------------------------===//
//
// Computes C[M,N] = A[M,K] * B[K,N]  (all row-major, fp16, fp32 accumulation)
//
// Uses the FP16 -> FP32 16x16x16 WMMA on RDNA3 (gfx11xx) and RDNA4
// (gfx1200/gfx1201).  Register-tiled: each wavefront computes a
// (KX*16) x (KY*16) output tile.  See wmma_frag.h for the per-arch
// fragment layout and the C/D row mapping conventions used below.

template<int KX, int KY>
__global__ void __launch_bounds__(32)
gemm_wmma_fp16_kernel(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int K, int N)
{
    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    int lane, sub;
    hip_wmma_lane_decode(threadIdx.x, &lane, &sub);

    const int m_base = m_tile * KX * kWmmaTile;
    const int n_base = n_tile * KY * kWmmaTile;

    constexpr int AB_SLOTS = hip_wmma_ab_slots();

    wmma_ab_frag_t a_frag[KX];
    wmma_ab_frag_t b_frag[KY];
    hip_float8 c_frag[KY][KX] = {};

    for (int k = 0; k < K; k += kWmmaTile) {
#pragma unroll
        for (int ele = 0; ele < AB_SLOTS; ++ele) {
            int kk = hip_wmma_ab_k_for_element(ele, sub);
#pragma unroll
            for (int x = 0; x < KX; ++x) {
                int row = m_base + x * kWmmaTile + lane;
                a_frag[x][ele] = (row < M) ? A[row * K + k + kk]
                                           : (_Float16)0;
            }
#pragma unroll
            for (int y = 0; y < KY; ++y) {
                int col = n_base + y * kWmmaTile + lane;
                b_frag[y][ele] = (col < N) ? B[(k + kk) * N + col]
                                           : (_Float16)0;
            }
        }

#pragma unroll
        for (int y = 0; y < KY; ++y)
#pragma unroll
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = hip_wmma_f32_16x16x16_f16(
                                    a_frag[x], b_frag[y], c_frag[y][x]);
    }

    for (int y = 0; y < KY; ++y) {
        for (int x = 0; x < KX; ++x) {
#pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + hip_wmma_cd_row(ele, sub);
                int c = n_base + y * kWmmaTile + lane;
                if (r < M && c < N)
                    C[r * N + c] = (_Float16)c_frag[y][x][ele];
            }
        }
    }
}

extern "C"
int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                       void* C, int M, int K, int N) {
    constexpr int KX = 2;
    constexpr int KY = 4;
    dim3 block(32);
    dim3 grid((N + KY * kWmmaTile - 1) / (KY * kWmmaTile),
              (M + KX * kWmmaTile - 1) / (KX * kWmmaTile));
    gemm_wmma_fp16_kernel<KX, KY><<<grid, block, 0, (hipStream_t)stream>>>(
        (const _Float16*)A, (const _Float16*)B, (_Float16*)C, M, K, N);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
