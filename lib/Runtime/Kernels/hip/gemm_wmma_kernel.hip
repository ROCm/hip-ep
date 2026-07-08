/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_custom_kernels.h"

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// The 16x16x16 fp16 WMMA builtin is an RDNA3-family (wave32) intrinsic. On GPU
// targets that do not provide it, this file must still COMPILE (it is built for
// every configured arch), but the WMMA path must never silently produce wrong
// results. Guard on __has_builtin (arch-agnostic -- no hard-coded arch name):
// where the intrinsic is missing, forward to a shim that traps, so a WMMA launch
// on an unsupported target aborts loudly instead of returning a zeroed tile. A
// target that needs WMMA on such an arch must add a real implementation here.
#if !defined(__has_builtin) ||                                                  \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_w32_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)                     \
  hipdnn_wmma_f32_16x16x16_f16_w32_unavailable((a), (b), (c))
#endif

static constexpr int kWmmaTile = 16;  // RDNA 3 WMMA tile dimension (16x16x16)

//===----------------------------------------------------------------------===//
// WMMA GEMM kernel for small-M fp16 MatMuls (M <= 512)
//===----------------------------------------------------------------------===//
//
// Computes C[M,N] = A[M,K] * B[K,N]  (all row-major, fp16, fp32 accumulation)
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
    const int lane = threadIdx.x % kWmmaTile;
    const int pair = threadIdx.x / kWmmaTile;

    const int m_base = m_tile * KX * kWmmaTile;
    const int n_base = n_tile * KY * kWmmaTile;

    half16 a_frag[KX];
    half16 b_frag[KY];
    float8 c_frag[KY][KX] = {};

    for (int k = 0; k < K; k += kWmmaTile) {
        for (int ele = 0; ele < kWmmaTile; ++ele) {
            for (int x = 0; x < KX; ++x) {
                int row = m_base + x * kWmmaTile + lane;
                a_frag[x][ele] = (row < M) ? A[row * K + k + ele]
                                            : (_Float16)0;
            }
            for (int y = 0; y < KY; ++y) {
                int col = n_base + y * kWmmaTile + lane;
                b_frag[y][ele] = (col < N) ? B[(k + ele) * N + col]
                                            : (_Float16)0;
            }
        }

        for (int y = 0; y < KY; ++y)
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    a_frag[x], b_frag[y], c_frag[y][x]);
    }

    for (int y = 0; y < KY; ++y) {
        for (int x = 0; x < KX; ++x) {
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + ele * 2 + pair;
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
