/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"

// half16 holds one WMMA A/B fragment; its vector width matches
// HIPDNN_WMMA_FRAG_ELEMS (16 on gfx11, 8 on the gfx12-style encoding
// gfx1170/gfx12xx use -- see hip_arch_compat.h) so it binds directly to
// whichever HIPDNN_WMMA_F32_16X16X16_F16 builtin is selected. The name is
// kept for continuity with the existing RDNA3 comments/callers even though
// the width is arch-dependent.
typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Where neither WMMA builtin is available (non-RDNA3/3.5/4 arch), trap
// instead so this file still compiles and a WMMA launch on an unsupported
// arch aborts loudly rather than returning garbage.
#if !HIPDNN_HAS_WMMA
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#undef HIPDNN_WMMA_F32_16X16X16_F16
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                                 \
  hipdnn_wmma_f32_16x16x16_f16_unavailable((a), (b), (c))
#endif

static constexpr int kWmmaTile = 16;  // WMMA tile dimension (16x16x16)

//===----------------------------------------------------------------------===//
// WMMA GEMM kernel for small-M fp16 MatMuls (M <= 512)
//===----------------------------------------------------------------------===//
//
// Computes C[M,N] = A[M,K] * B[K,N]  (all row-major, fp16, fp32 accumulation)
//
// Uses HIPDNN_WMMA_F32_16X16X16_F16 (hip_arch_compat.h) for hardware-
// accelerated 16x16x16 tile multiply-accumulate on RDNA 3/3.5/4.
//
// Register-tiled: each wavefront computes a (KX*16) x (KY*16) output tile.
// KX A fragments are each reused KY times; KY B fragments each reused KX times.
// Grid: (ceil(N/(KY*16)), ceil(M/(KX*16)))
//
// WMMA fragment layout (wave32; two encodings, selected at compile time by
// HIPDNN_WMMA_GFX12 -- see hip_arch_compat.h for how each was verified):
//   gfx11 (HIPDNN_WMMA_FRAG_ELEMS==16): a_frag[k] at lane m = A_tile[m, k];
//     b_frag[k] at lane n = B_tile[k, n]. Lanes 0-15 and 16-31 hold identical
//     data (required by this encoding), satisfied by indexing with
//     threadIdx.x % 16 == m (or n).
//   gfx12-style (HIPDNN_WMMA_FRAG_ELEMS==8, e.g. gfx1170/gfx12xx): no
//     replication; a_frag[j] at lane L = A_tile[L%16, hipdnn_wmma_k_off(pair)+j],
//     b_frag[j] at lane L = B_tile[hipdnn_wmma_k_off(pair)+j, L%16], where
//     pair = L/16 selects which half of K this lane contributes.
//   c_frag (float8, both encodings): c_frag[ele] -> row hipdnn_wmma_acc_row(pair, ele),
//     col (lane%16). lane = threadIdx.x % 16, pair = threadIdx.x / 16.

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

    // K-axis base this lane's pair-half contributes: 0 on gfx11 (each lane
    // loads the full 16-wide K range), pair*8 on the gfx12-style encoding
    // (each lane loads only its half, no replication).
    const int k_off = hipdnn_wmma_k_off(pair);

    for (int k = 0; k < K; k += kWmmaTile) {
        for (int ele = 0; ele < HIPDNN_WMMA_FRAG_ELEMS; ++ele) {
            for (int x = 0; x < KX; ++x) {
                int row = m_base + x * kWmmaTile + lane;
                a_frag[x][ele] = (row < M) ? A[row * K + k + k_off + ele]
                                            : (_Float16)0;
            }
            for (int y = 0; y < KY; ++y) {
                int col = n_base + y * kWmmaTile + lane;
                b_frag[y][ele] = (col < N) ? B[(k + k_off + ele) * N + col]
                                            : (_Float16)0;
            }
        }

        for (int y = 0; y < KY; ++y)
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = HIPDNN_WMMA_F32_16X16X16_F16(
                                    a_frag[x], b_frag[y], c_frag[y][x]);
    }

    for (int y = 0; y < KY; ++y) {
        for (int x = 0; x < KX; ++x) {
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + hipdnn_wmma_acc_row(pair, ele);
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
    // WMMA is RDNA-only (wave32); CDNA (wave64, e.g. MI300/MI350) has no WMMA
    // and would otherwise hit the __builtin_trap() fallback above at launch.
    if (hipdnn_device_is_wave64())
        return -1;
    constexpr int KX = 2;
    constexpr int KY = 4;
    dim3 block(32);
    dim3 grid((N + KY * kWmmaTile - 1) / (KY * kWmmaTile),
              (M + KX * kWmmaTile - 1) / (KX * kWmmaTile));
    gemm_wmma_fp16_kernel<KX, KY><<<grid, block, 0, (hipStream_t)stream>>>(
        (const _Float16*)A, (const _Float16*)B, (_Float16*)C, M, K, N);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
