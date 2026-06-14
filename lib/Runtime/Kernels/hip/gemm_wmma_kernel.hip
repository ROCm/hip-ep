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
// FullTile == true is launched only when M is a multiple of KX*16 and N is a
// multiple of KY*16, so every (row, col) this wave touches is in-bounds. That
// lets the compiler drop all per-element edge checks on the load and store
// paths, producing branch-free vector loads/stores. FullTile == false keeps the
// guarded path for remainder tiles.
template<int KX, int KY, bool TransB, bool FullTile>
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
            if (FullTile || row < M) {
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
                if (FullTile || col < N) {
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
                    b_frag[y][ele] = (FullTile || col < N) ? B[(k + ele) * N + col]
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
        float bv = (bias && (FullTile || c < N)) ? (float)bias[c] : 0.0f;
        #pragma unroll
        for (int x = 0; x < KX; ++x) {
            #pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + ele * 2 + pair;
                if (FullTile || (r < M && c < N))
                    C[r * N + c] = (_Float16)(c_frag[y][x][ele] + bv);
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// LDS-staged multi-wave variant (transB == 1 only)
//===----------------------------------------------------------------------===//
//
// The single-wave kernel above re-reads the same A row-panel from global memory
// once per N-tile and the same B col-panel once per M-tile, so a large GEMM
// streams A and B from global many times. This variant groups WAVES_M x WAVES_N
// waves into one block that cooperatively stages a single A panel [BM x 16] and
// B panel [BN x 16] into LDS each k-step; every wave then reads its WMMA
// fragments from LDS instead of global. Each k-panel is therefore fetched from
// global exactly once per block instead of once per wave-tile.
//
// Numerics are bit-identical to the single-wave kernel: the per-wave a_frag /
// b_frag index derivation and the WMMA accumulation order are unchanged; only
// the *source* of the fragments (LDS vs global) differs. Restricted to
// transB == 1 (B is [N,K], the nn.Linear weight layout) so the B panel is
// contiguous along K and vector-loadable.
//
//   BM = WAVES_M * KX * 16   (block tile rows)
//   BN = WAVES_N * KY * 16   (block tile cols)
template <int KX, int KY, int WAVES_M, int WAVES_N, bool FullTile>
__global__ void __launch_bounds__(WAVES_M* WAVES_N * 32)
gemm_wmma_fp16_kernel_lds(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    const _Float16* __restrict__ bias,
    int M, int K, int N)
{
    constexpr int TPB = WAVES_M * WAVES_N * 32;
    constexpr int BM = WAVES_M * KX * kWmmaTile;
    constexpr int BN = WAVES_N * KY * kWmmaTile;

    __shared__ _Float16 As[BM][kWmmaTile];
    __shared__ _Float16 Bs[BN][kWmmaTile];

    const int tid = threadIdx.x;
    const int wid = tid / 32;          // wave index within block
    const int lane = (tid % 32) % kWmmaTile;
    const int pair = (tid % 32) / kWmmaTile;
    const int wave_m = wid / WAVES_N;
    const int wave_n = wid % WAVES_N;

    const int m_block = blockIdx.y * BM;
    const int n_block = blockIdx.x * BN;
    const int m_base = m_block + wave_m * KX * kWmmaTile;
    const int n_base = n_block + wave_n * KY * kWmmaTile;

    half16 a_frag[KX];
    half16 b_frag[KY];
    float8 c_frag[KY][KX] = {};

    for (int k = 0; k < K; k += kWmmaTile) {
        // Cooperatively stage the A panel [BM x 16] into LDS.
        #pragma unroll
        for (int i = tid; i < BM; i += TPB) {
            int row = m_block + i;
            if (FullTile || row < M) {
                *reinterpret_cast<half16*>(&As[i][0]) =
                    *reinterpret_cast<const half16*>(&A[row * K + k]);
            } else {
                #pragma unroll
                for (int ele = 0; ele < kWmmaTile; ++ele)
                    As[i][ele] = (_Float16)0;
            }
        }
        // Cooperatively stage the B panel [BN x 16] into LDS (transB == 1:
        // B[col, k] is contiguous along K).
        #pragma unroll
        for (int i = tid; i < BN; i += TPB) {
            int col = n_block + i;
            if (FullTile || col < N) {
                *reinterpret_cast<half16*>(&Bs[i][0]) =
                    *reinterpret_cast<const half16*>(&B[col * K + k]);
            } else {
                #pragma unroll
                for (int ele = 0; ele < kWmmaTile; ++ele)
                    Bs[i][ele] = (_Float16)0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int x = 0; x < KX; ++x)
            a_frag[x] = *reinterpret_cast<const half16*>(
                &As[wave_m * KX * kWmmaTile + x * kWmmaTile + lane][0]);
        #pragma unroll
        for (int y = 0; y < KY; ++y)
            b_frag[y] = *reinterpret_cast<const half16*>(
                &Bs[wave_n * KY * kWmmaTile + y * kWmmaTile + lane][0]);

        #pragma unroll
        for (int y = 0; y < KY; ++y)
            #pragma unroll
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                   a_frag[x], b_frag[y], c_frag[y][x]);

        __syncthreads();  // protect LDS before the next k-panel overwrites it
    }

    #pragma unroll
    for (int y = 0; y < KY; ++y) {
        int c = n_base + y * kWmmaTile + lane;
        float bv = (bias && (FullTile || c < N)) ? (float)bias[c] : 0.0f;
        #pragma unroll
        for (int x = 0; x < KX; ++x) {
            #pragma unroll
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + ele * 2 + pair;
                if (FullTile || (r < M && c < N))
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
    auto A16 = (const _Float16*)A;
    auto B16 = (const _Float16*)B;
    auto O16 = (_Float16*)output;

    if (transB) {
        // transB == 1: use the LDS-staged multi-wave kernel, which cuts
        // redundant global reads of the A/B panels. Block tile BM x BN with
        // WAVES_M x WAVES_N waves.
        constexpr int WAVES_M = 2;
        constexpr int WAVES_N = 4;
        constexpr int BM = WAVES_M * KX * kWmmaTile;  // 64
        constexpr int BN = WAVES_N * KY * kWmmaTile;  // 128
        dim3 block(WAVES_M * WAVES_N * 32);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        const bool full_tile = (M % BM == 0) && (N % BN == 0);
        if (full_tile)
            gemm_wmma_fp16_kernel_lds<KX, KY, WAVES_M, WAVES_N, true><<<grid, block, 0, (hipStream_t)stream>>>(A16, B16, O16, bias, M, K, N);
        else
            gemm_wmma_fp16_kernel_lds<KX, KY, WAVES_M, WAVES_N, false><<<grid, block, 0, (hipStream_t)stream>>>(A16, B16, O16, bias, M, K, N);
    } else {
        // transB == 0: keep the single-wave global-load kernel.
        dim3 block(32);
        dim3 grid((N + KY * kWmmaTile - 1) / (KY * kWmmaTile),
                  (M + KX * kWmmaTile - 1) / (KX * kWmmaTile));
        const bool full_tile = (M % (KX * kWmmaTile) == 0) && (N % (KY * kWmmaTile) == 0);
        if (full_tile)
            gemm_wmma_fp16_kernel<KX, KY, false, true><<<grid, block, 0, (hipStream_t)stream>>>(A16, B16, O16, bias, M, K, N);
        else
            gemm_wmma_fp16_kernel<KX, KY, false, false><<<grid, block, 0, (hipStream_t)stream>>>(A16, B16, O16, bias, M, K, N);
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
