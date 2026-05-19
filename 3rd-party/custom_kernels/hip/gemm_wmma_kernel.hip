/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

 #include <hip/hip_runtime.h>
 #include <hip/hip_fp16.h>
 
 #include "hip_custom_kernels.h"
 
 // Model DLLs link this static lib with /MT (static CRT), so std::getenv
 // cannot see env vars set by the host EP process. Use the Win32 API for
 // the opt-in env gates below (mirrors lib/Runtime/debug_log.h).
 #ifdef _WIN32
 extern "C" __declspec(dllimport)
 unsigned long __stdcall GetEnvironmentVariableA(
     const char*, char*, unsigned long);
 #endif
 
 typedef _Float16 half16 __attribute__((ext_vector_type(16)));
 typedef float float8 __attribute__((ext_vector_type(8)));
 
 static constexpr int kWmmaTile = 16;  // RDNA 3 WMMA tile dimension (16x16x16)
 
 //===----------------------------------------------------------------------===//
 // WMMA GEMM kernel — fp16 in / fp32 accumulate / fp16 out
 //===----------------------------------------------------------------------===//
 //
 // Computes C[M,N] = A[M,K] * B[K,N]  (all row-major). No alpha/beta/bias.
 //
 // Ported from gemm-benchmarks gemm_wmma_fp16_lds_v3_kernel — v2's double-
 // buffered LDS staging plus a block-row-grouped swizzle that walks GROUP_M
 // consecutive block-rows in a strip before advancing to the next strip, so the
 // B-column panel stays hot in L2 across the strip. Falls back to the v2 linear
 // ordering automatically when gridDim.y < GROUP_M or gridDim.y % GROUP_M != 0,
 // so degenerate shapes (e.g. decode M < BM) get no regression.
 //   * 256 threads (8 waves of 32) per block; gives enough outstanding waves to
 //     hide HBM and WMMA latency on RDNA3.
 //   * 128 x 128 output tile per block, partitioned 4 (M) x 2 (N) over waves so
 //     each wave owns a 32 x 64 sub-tile = 2 x 4 WMMA fragments = 8 float8
 //     accumulators per lane (~64 VGPRs, under the spill threshold).
 //   * Double-buffered LDS: As[2][BM][BK+pad], Bs[2][BK][BN+pad]. The HBM->LDS
 //     prefetch for the next K-block runs concurrently with the WMMA on the
 //     current buffer, so HBM bandwidth is hidden behind compute.
 //   * +4 fp16 inner-dim pad on LDS dodges 32-bank conflicts on the fragment
 //     load path (each wave's 16 lanes hit different banks).
 //   * float4 vectorized HBM loads (8 fp16 / thread / load). Each 128x16 A tile
 //     and 16x128 B tile = 2048 halfs, exactly 256 threads * 8 halfs.
 //   * M/N/K tail handling preserved from v2 — scalar fallback in load_AB, masked
 //     store. Eligible shapes are still gated upstream by wrap_gemm (fp16,
 //     no transpose, no bias, alpha=1, K % 16 == 0 && N % 16 == 0).
 //   * GROUP_M = 2: working-set of one B panel (K*BN*2 B) plus 2 A panels stays
 //     within Strix Halo's 4 MB L2 for K up to ~4096; larger K loses the L2
 //     reuse benefit but the kernel still runs correctly.
 //
 // WMMA fragment layout (wave32):
 //   a_frag (half16): a_frag[k] at lane m = A_tile[m, k]
 //   b_frag (half16): b_frag[k] at lane n = B_tile[k, n]
 //   c_frag (float8): c_frag[ele] -> row (ele*2 + pair), col (lane)
 //                    where lane = threadIdx.x % 16, pair = threadIdx.x / 16
 // Lanes 0-15 and 16-31 load identical data (required by RDNA 3 WMMA).
 
 __global__ __launch_bounds__(256) void
 gemm_wmma_fp16_kernel(
     const _Float16* __restrict__ A,
     const _Float16* __restrict__ B,
     _Float16*       __restrict__ C,
     int M, int K, int N)
 {
     constexpr int BM           = 128;
     constexpr int BN           = 128;
     constexpr int BK           = 16;
     constexpr int LDS_PAD      = 4;
     constexpr int WAVE_GRID_M  = 4;                       // 4 waves cover BM
     constexpr int WAVE_GRID_N  = 2;                       // 2 waves cover BN
     constexpr int WAVE_TILE_M  = BM / WAVE_GRID_M;        // 32
     constexpr int WAVE_TILE_N  = BN / WAVE_GRID_N;        // 64
     constexpr int WAVE_M_TILES = WAVE_TILE_M / kWmmaTile; // 2
     constexpr int WAVE_N_TILES = WAVE_TILE_N / kWmmaTile; // 4
     constexpr int GROUP_M      = 2;                       // L2-reuse strip width
 
     __shared__ _Float16 As[2][BM][BK + LDS_PAD];
     __shared__ _Float16 Bs[2][BK][BN + LDS_PAD];
 
     // Block swizzle: walk GROUP_M block-rows in a column-major strip so the
     // B-column panel stays hot in L2 for GROUP_M * gridDim.x blocks instead of
     // just gridDim.x. Degenerate grids fall back to the v2 linear ordering.
     int bx, by;
     if (gridDim.y < GROUP_M || (gridDim.y % GROUP_M) != 0) {
         bx = blockIdx.x;
         by = blockIdx.y;
     } else {
         const int gN             = gridDim.x;
         const int linear         = blockIdx.y * gN + blockIdx.x;
         const int blocks_per_grp = GROUP_M * gN;
         const int grp            = linear / blocks_per_grp;
         const int in_grp         = linear - grp * blocks_per_grp;
         const int local_by       = in_grp % GROUP_M;
         const int local_bx       = in_grp / GROUP_M;
         by = grp * GROUP_M + local_by;
         bx = local_bx;
     }
 
     const int tid     = threadIdx.x;          // 0..255
     const int waveId  = tid / 32;             // 0..7
     const int waveRow = waveId % WAVE_GRID_M; // 0..3 (M wave)
     const int waveCol = waveId / WAVE_GRID_M; // 0..1 (N wave)
     const int lane    = tid % 32;
     const int laneM   = lane % kWmmaTile;     // 0..15
     const int pair    = lane / kWmmaTile;     // 0 or 1
 
     const int blockM = by * BM;
     const int blockN = bx * BN;
     const int waveM0 = waveRow * WAVE_TILE_M;
     const int waveN0 = waveCol * WAVE_TILE_N;
 
     float8 c_frag[WAVE_M_TILES][WAVE_N_TILES] = {};
 
     // ---- Cooperative HBM -> LDS load (one K-block worth) ------------------
     // A tile: 128 rows * 16 K = 2048 halfs; 256 thr * 8 halfs/thr = 2048.
     // B tile: 16 rows * 128 N = 2048 halfs; same.
     auto load_AB = [&](int kStep, int buf) {
         {
             int row  = tid / 2;
             int col0 = (tid % 2) * 8;
             int gr   = blockM + row;
             int gc   = kStep  + col0;
             float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
             if (gr < M && gc + 7 < K) {
                 v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
             } else if (gr < M) {
                 _Float16 tmp[8] = {};
 #pragma unroll
                 for (int i = 0; i < 8; ++i) {
                     int cc = gc + i;
                     if (cc < K) tmp[i] = A[gr * K + cc];
                 }
                 v = *reinterpret_cast<float4*>(tmp);
             }
             *reinterpret_cast<float4*>(&As[buf][row][col0]) = v;
         }
         {
             int row  = tid / 16;          // 0..15
             int col0 = (tid % 16) * 8;    // 0,8,..,120
             int gr   = kStep  + row;
             int gc   = blockN + col0;
             float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
             if (gr < K && gc + 7 < N) {
                 v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
             } else if (gr < K) {
                 _Float16 tmp[8] = {};
 #pragma unroll
                 for (int i = 0; i < 8; ++i) {
                     int cc = gc + i;
                     if (cc < N) tmp[i] = B[gr * N + cc];
                 }
                 v = *reinterpret_cast<float4*>(tmp);
             }
             *reinterpret_cast<float4*>(&Bs[buf][row][col0]) = v;
         }
     };
 
     // ---- Prologue: prime buffer 0 -----------------------------------------
     load_AB(0, 0);
     __syncthreads();
 
     int buf = 0;
     for (int kStep = 0; kStep < K; kStep += BK) {
         const int next_kStep = kStep + BK;
         const int next_buf   = buf ^ 1;
         const bool has_next  = (next_kStep < K);
 
         // Issue HBM -> LDS prefetch for next K-block into the OTHER buffer.
         // These global loads stay in flight while the WMMA's below execute.
         if (has_next) {
             load_AB(next_kStep, next_buf);
         }
 
         // ---- Bring fragments from current buf into registers --------------
         half16 a_frag[WAVE_M_TILES];
         half16 b_frag[WAVE_N_TILES];
 
 #pragma unroll
         for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
             int m_local = waveM0 + wm * kWmmaTile + laneM;
 #pragma unroll
             for (int ele = 0; ele < kWmmaTile; ++ele) {
                 a_frag[wm][ele] = As[buf][m_local][ele];
             }
         }
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             int n_local = waveN0 + wn * kWmmaTile + laneM;
 #pragma unroll
             for (int ele = 0; ele < kWmmaTile; ++ele) {
                 b_frag[wn][ele] = Bs[buf][ele][n_local];
             }
         }
 
         // ---- 8 WMMA ops per wave (2 M-tiles * 4 N-tiles) -------------------
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
 #pragma unroll
             for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
                 c_frag[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                     a_frag[wm], b_frag[wn], c_frag[wm][wn]);
             }
         }
 
         // Wait for the prefetch into next_buf to land before next iter reads.
         __syncthreads();
         buf = next_buf;
     }
 
     // ---- Store ------------------------------------------------------------
 #pragma unroll
     for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             int m_base = blockM + waveM0 + wm * kWmmaTile;
             int n_base = blockN + waveN0 + wn * kWmmaTile;
             int c      = n_base + laneM;
 #pragma unroll
             for (int ele = 0; ele < 8; ++ele) {
                 int r = m_base + ele * 2 + pair;
                 if (r < M && c < N) {
                     C[r * N + c] = (_Float16)c_frag[wm][wn][ele];
                 }
             }
         }
     }
 }
 
 //===----------------------------------------------------------------------===//
 // GEMV-style kernel for small M (decode / speculative batch)
 //===----------------------------------------------------------------------===//
 //
 // The v3 WMMA kernel above uses BM=128. For M=1 (single-token decode) that
 // wastes 127/128 of every WMMA fragment. This kernel skips WMMA entirely:
 //   * Each block owns BN consecutive output columns (BN = blockDim.x).
 //   * Each thread t owns ONE output column n = blockIdx.x * BN + t.
 //   * In the K loop, every thread reads one fp16 from B[k, n]. Across the
 //     warp those reads are adjacent columns at the same row, so they
 //     coalesce to one 256-byte transaction per warp per K-step.
 //   * A is broadcast across threads (same A[m, k] per K-step), so L1
 //     services it; no LDS staging needed.
 //
 // Memory cost per kernel (M=1, N=4096, K=4096):
 //   * B traffic: K * N * 2 = 32 MB (unavoidable -- this is the whole matrix).
 //   * A traffic: M * K * 2 = 8 KB (broadcast across all blocks, L1-cached).
 // Performance is bandwidth-bound; the kernel just needs to keep enough
 // loads in flight to saturate HBM. Removing LDS staging and the inner
 // __syncthreads from earlier versions was the critical win -- 4x faster
 // than an LDS-staged variant on M=1 K=4096 shapes.
 //
 // Template parameter M_MAX is the unrolled bound; the kernel honors the
 // runtime M argument (loops only over rows 0..M-1). Launcher picks M_MAX
 // in {1, 2, 4}.
 template<int M_MAX, int BN>
 __global__ __launch_bounds__(BN)
 void gemm_gemv_fp16_kernel(
     const _Float16* __restrict__ A,  // [M, K] row-major
     const _Float16* __restrict__ B,  // [K, N] row-major
     _Float16* __restrict__ C,        // [M, N] row-major
     int M, int K, int N)
 {
     const int tid = threadIdx.x;
     const int n   = blockIdx.x * BN + tid;
     if (n >= N) return;
 
     float acc[M_MAX];
 #pragma unroll
     for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;
 
     // Unrolled K loop. UNROLL=8 chosen empirically: smaller leaves too few
     // independent loads in flight to hide HBM latency; larger (UNROLL=16)
     // bloats live B[] across the FMA stage and costs occupancy on big-N
     // shapes (measured 1x4096x14336: 1.4ms@U8 vs 2.05ms@U16).
     constexpr int UNROLL = 8;
     int k = 0;
     for (; k + UNROLL <= K; k += UNROLL) {
         float b[UNROLL];
 #pragma unroll
         for (int u = 0; u < UNROLL; ++u) {
             b[u] = (float)B[(k + u) * N + n];
         }
 #pragma unroll
         for (int m = 0; m < M_MAX; ++m) {
             if (m < M) {
 #pragma unroll
                 for (int u = 0; u < UNROLL; ++u) {
                     acc[m] += (float)A[m * K + (k + u)] * b[u];
                 }
             }
         }
     }
     // Tail (K not a multiple of UNROLL).
     for (; k < K; ++k) {
         float b_val = (float)B[k * N + n];
 #pragma unroll
         for (int m = 0; m < M_MAX; ++m) {
             if (m < M) acc[m] += (float)A[m * K + k] * b_val;
         }
     }
 
 #pragma unroll
     for (int m = 0; m < M_MAX; ++m) {
         if (m < M) C[m * N + n] = (_Float16)acc[m];
     }
 }
 
 static int launch_gemv(hipStream_t stream, const _Float16* A,
                        const _Float16* B, _Float16* C,
                        int M, int K, int N) {
     constexpr int BN = 128;
     dim3 block(BN);
     dim3 grid((N + BN - 1) / BN);
     if (M <= 1) {
         gemm_gemv_fp16_kernel<1, BN>
             <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
     } else if (M <= 2) {
         gemm_gemv_fp16_kernel<2, BN>
             <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
     } else {
         gemm_gemv_fp16_kernel<4, BN>
             <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
     }
     return hipGetLastError() == hipSuccess ? 0 : -1;
 }
 
 //===----------------------------------------------------------------------===//
 // BM=16 WMMA kernel for the gap between GEMV (M <= 4) and v3 (BM=128)
 //===----------------------------------------------------------------------===//
 //
 // v3 always allocates 128 M-rows per block. For batch-decode shapes
 // (M = 8..32, e.g. speculative decode, small chunked prefill) that wastes
 // (128-M)/128 = 75-94% of every WMMA fragment. This kernel keeps the same
 // LDS-staged design but with BM=16 -- exactly one WMMA M-tile per block --
 // so utilization stays near 100% across the M dimension. The launcher uses
 // it for 4 < M <= 32 by default; M >= 33 falls back to v3 (where the
 // 128x128 tile starts amortizing its compute density advantage).
 //
 // Geometry: BM=16, BN=128, BK=16, 128 threads (4 waves), wave grid 1 x 4
 // (each wave owns 16 rows by 32 cols = 1 M-tile x 2 N-tiles = 2 frags).
 // LDS: As[2][16][20] + Bs[2][16][132] ~= 9.7 KB per block, plenty of room
 // for high occupancy on Strix Halo's 64 KB / CU LDS budget.
 
 __global__ __launch_bounds__(128) void
 gemm_wmma_fp16_bm16_kernel(
     const _Float16* __restrict__ A,
     const _Float16* __restrict__ B,
     _Float16*       __restrict__ C,
     int M, int K, int N)
 {
     constexpr int BM           = 16;
     constexpr int BN           = 128;
     constexpr int BK           = 16;
     constexpr int LDS_PAD      = 4;
     constexpr int WAVE_GRID_N  = 4;
     constexpr int WAVE_TILE_N  = BN / WAVE_GRID_N;        // 32
     constexpr int WAVE_N_TILES = WAVE_TILE_N / kWmmaTile; // 2
 
     __shared__ _Float16 As[2][BM][BK + LDS_PAD];
     __shared__ _Float16 Bs[2][BK][BN + LDS_PAD];
 
     const int tid     = threadIdx.x;          // 0..127
     const int waveId  = tid / 32;             // 0..3 (only N-direction waves)
     const int lane    = tid % 32;
     const int laneM   = lane % kWmmaTile;     // 0..15
     const int pair    = lane / kWmmaTile;     // 0 or 1
 
     const int bx = blockIdx.x;
     const int by = blockIdx.y;
     const int blockM = by * BM;
     const int blockN = bx * BN;
     const int waveN0 = waveId * WAVE_TILE_N;
 
     float8 c_frag[WAVE_N_TILES] = {};
 
     // Cooperative HBM -> LDS load.
     //   A tile: 16 rows * 16 K = 256 halfs. 128 threads * 2 halfs = 1 float / thr.
     //   B tile: 16 rows * 128 N = 2048 halfs. 128 threads * 16 halfs = 2 float4 / thr.
     auto load_AB = [&](int kStep, int buf) {
         // A: one float (2 halfs) per thread, [row=tid/8, col=2*(tid%8)]
         {
             int row  = tid / 8;        // 0..15
             int col0 = (tid % 8) * 2;  // 0,2,...,14
             int gr   = blockM + row;
             int gc   = kStep  + col0;
             float v = 0.0f;
             if (gr < M && gc + 1 < K) {
                 v = *reinterpret_cast<const float*>(&A[gr * K + gc]);
             } else if (gr < M) {
                 _Float16 tmp[2] = {};
 #pragma unroll
                 for (int i = 0; i < 2; ++i) {
                     int cc = gc + i;
                     if (cc < K) tmp[i] = A[gr * K + cc];
                 }
                 v = *reinterpret_cast<float*>(tmp);
             }
             *reinterpret_cast<float*>(&As[buf][row][col0]) = v;
         }
         // B: 256 float4 cells in a 16-row x 16-float4-col grid; 128 threads x 2 cells.
 #pragma unroll
         for (int pass = 0; pass < 2; ++pass) {
             int i = tid + pass * 128;        // 0..255
             int row  = i / 16;               // 0..15
             int col0 = (i % 16) * 8;         // 0,8,...,120
             int gr   = kStep  + row;
             int gc   = blockN + col0;
             float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
             if (gr < K && gc + 7 < N) {
                 v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
             } else if (gr < K) {
                 _Float16 tmp[8] = {};
 #pragma unroll
                 for (int j = 0; j < 8; ++j) {
                     int cc = gc + j;
                     if (cc < N) tmp[j] = B[gr * N + cc];
                 }
                 v = *reinterpret_cast<float4*>(tmp);
             }
             *reinterpret_cast<float4*>(&Bs[buf][row][col0]) = v;
         }
     };
 
     load_AB(0, 0);
     __syncthreads();
 
     int buf = 0;
     for (int kStep = 0; kStep < K; kStep += BK) {
         const int next_kStep = kStep + BK;
         const int next_buf   = buf ^ 1;
         const bool has_next  = (next_kStep < K);
 
         if (has_next) {
             load_AB(next_kStep, next_buf);
         }
 
         // Fragment loads: BM=16 means a single A-tile per wave.
         half16 a_frag;
         half16 b_frag[WAVE_N_TILES];
 
 #pragma unroll
         for (int ele = 0; ele < kWmmaTile; ++ele) {
             a_frag[ele] = As[buf][laneM][ele];
         }
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             int n_local = waveN0 + wn * kWmmaTile + laneM;
 #pragma unroll
             for (int ele = 0; ele < kWmmaTile; ++ele) {
                 b_frag[wn][ele] = Bs[buf][ele][n_local];
             }
         }
 
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             c_frag[wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                             a_frag, b_frag[wn], c_frag[wn]);
         }
 
         __syncthreads();
         buf = next_buf;
     }
 
     // Store.
 #pragma unroll
     for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
         int n_base = blockN + waveN0 + wn * kWmmaTile;
         int c      = n_base + laneM;
 #pragma unroll
         for (int ele = 0; ele < 8; ++ele) {
             int r = blockM + ele * 2 + pair;
             if (r < M && c < N) {
                 C[r * N + c] = (_Float16)c_frag[wn][ele];
             }
         }
     }
 }
 
 static int launch_bm16(hipStream_t stream, const _Float16* A,
                        const _Float16* B, _Float16* C,
                        int M, int K, int N) {
     constexpr int BM = 16;
     constexpr int BN = 128;
     dim3 block(128);
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     gemm_wmma_fp16_bm16_kernel<<<grid, block, 0, stream>>>(
         A, B, C, M, K, N);
     return hipGetLastError() == hipSuccess ? 0 : -1;
 }
 
 //===----------------------------------------------------------------------===//
 // Split-K WMMA kernel + reduce, for small-K prefill where v3 underutilizes
 //===----------------------------------------------------------------------===//
 //
 // v3 launches one block per (BM, BN) tile and that block walks the entire K
 // dimension serially. At small K (~2048) each block finishes quickly and the
 // grid retires before the CUs can hide instruction-fetch / scheduling cost,
 // so total time is dominated by per-block startup rather than throughput.
 // Split-K trades a tiny extra reduce pass for K_SPLITS times more blocks:
 // each phase-1 block does K/K_SPLITS iterations, then the reduce kernel
 // sums the K_SPLITS partial outputs into the final fp16 C tensor.
 //
 // Workspace: K_SPLITS * M * N float partials (fp32 to avoid precision loss
 // across the partial sums). Allocated lazily via a per-process static cache
 // that grows on demand and never shrinks; freed implicitly at process exit.
 // This is single-stream-safe (the EP serializes inference on one HIP stream
 // per RuntimeState) but NOT multi-stream-safe -- two streams calling this
 // concurrently would race on the workspace. The same caveat applies to the
 // EP's existing hipblasLt workspace, so it's consistent.
 //
 // Phase-1 kernel is structurally v2/v3 with two changes:
 //   1. blockIdx.z selects the K-split, so each block walks K range
 //      [kid * K/K_SPLITS, (kid+1) * K/K_SPLITS).
 //   2. Output writes are fp32 stores into workspace[kid * M*N + r * N + c]
 //      instead of fp16 stores into C. No swizzle -- the L2-reuse trick is
 //      neutralized once K is small enough to trigger this path.
 //
 // EXPERIMENTAL: as of 2026-05 split-K did NOT win on the LLM sweep shapes
 // on gfx1151. Tested K_SPLITS in {2, 4} across 512x2048x8192 (Llama-1B
 // FFN-up, v3 6.45ms -> splitK=2 6.5ms / splitK=4 10.7ms), 512x2880x2880
 // (gpt-oss-20b attn-out, v3 1.35ms -> splitK=2 1.5ms), 2048x2048x2048
 // (v3 1.7ms -> splitK=2 2.45ms). The fp32-partial writes and the reduce
 // pass cost more than the per-block scheduling savings on this arch.
 // Dispatch is therefore gated behind HIPDNN_EP_GEMM_WMMA_SPLITK=1 for
 // further experimentation -- not used in the default path.
 template<int K_SPLITS>
 __global__ __launch_bounds__(256) void
 gemm_wmma_fp16_splitk_phase1_kernel(
     const _Float16* __restrict__ A,
     const _Float16* __restrict__ B,
     float*          __restrict__ workspace,   // [K_SPLITS, M, N]
     int M, int K, int N)
 {
     constexpr int BM           = 128;
     constexpr int BN           = 128;
     constexpr int BK           = 16;
     constexpr int LDS_PAD      = 4;
     constexpr int WAVE_GRID_M  = 4;
     constexpr int WAVE_GRID_N  = 2;
     constexpr int WAVE_TILE_M  = BM / WAVE_GRID_M;
     constexpr int WAVE_TILE_N  = BN / WAVE_GRID_N;
     constexpr int WAVE_M_TILES = WAVE_TILE_M / kWmmaTile;
     constexpr int WAVE_N_TILES = WAVE_TILE_N / kWmmaTile;
 
     __shared__ _Float16 As[2][BM][BK + LDS_PAD];
     __shared__ _Float16 Bs[2][BK][BN + LDS_PAD];
 
     const int k_split_id   = blockIdx.z;
     const int K_per_split  = (K + K_SPLITS - 1) / K_SPLITS;
     const int k_start      = k_split_id * K_per_split;
     int k_end_tmp          = k_start + K_per_split;
     if (k_end_tmp > K) k_end_tmp = K;
     const int k_end        = k_end_tmp;
     if (k_start >= k_end) return;  // this split is empty
 
     const int tid     = threadIdx.x;
     const int waveId  = tid / 32;
     const int waveRow = waveId % WAVE_GRID_M;
     const int waveCol = waveId / WAVE_GRID_M;
     const int lane    = tid % 32;
     const int laneM   = lane % kWmmaTile;
     const int pair    = lane / kWmmaTile;
 
     const int bx = blockIdx.x;
     const int by = blockIdx.y;
     const int blockM = by * BM;
     const int blockN = bx * BN;
     const int waveM0 = waveRow * WAVE_TILE_M;
     const int waveN0 = waveCol * WAVE_TILE_N;
 
     float8 c_frag[WAVE_M_TILES][WAVE_N_TILES] = {};
 
     auto load_AB = [&](int kStep, int buf) {
         {
             int row  = tid / 2;
             int col0 = (tid % 2) * 8;
             int gr   = blockM + row;
             int gc   = kStep  + col0;
             float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
             if (gr < M && gc + 7 < K) {
                 v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
             } else if (gr < M) {
                 _Float16 tmp[8] = {};
 #pragma unroll
                 for (int i = 0; i < 8; ++i) {
                     int cc = gc + i;
                     if (cc < K) tmp[i] = A[gr * K + cc];
                 }
                 v = *reinterpret_cast<float4*>(tmp);
             }
             *reinterpret_cast<float4*>(&As[buf][row][col0]) = v;
         }
         {
             int row  = tid / 16;
             int col0 = (tid % 16) * 8;
             int gr   = kStep  + row;
             int gc   = blockN + col0;
             float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
             if (gr < K && gc + 7 < N) {
                 v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
             } else if (gr < K) {
                 _Float16 tmp[8] = {};
 #pragma unroll
                 for (int i = 0; i < 8; ++i) {
                     int cc = gc + i;
                     if (cc < N) tmp[i] = B[gr * N + cc];
                 }
                 v = *reinterpret_cast<float4*>(tmp);
             }
             *reinterpret_cast<float4*>(&Bs[buf][row][col0]) = v;
         }
     };
 
     load_AB(k_start, 0);
     __syncthreads();
 
     int buf = 0;
     for (int kStep = k_start; kStep < k_end; kStep += BK) {
         const int next_kStep = kStep + BK;
         const int next_buf   = buf ^ 1;
         const bool has_next  = (next_kStep < k_end);
 
         if (has_next) {
             load_AB(next_kStep, next_buf);
         }
 
         half16 a_frag[WAVE_M_TILES];
         half16 b_frag[WAVE_N_TILES];
 
 #pragma unroll
         for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
             int m_local = waveM0 + wm * kWmmaTile + laneM;
 #pragma unroll
             for (int ele = 0; ele < kWmmaTile; ++ele) {
                 a_frag[wm][ele] = As[buf][m_local][ele];
             }
         }
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             int n_local = waveN0 + wn * kWmmaTile + laneM;
 #pragma unroll
             for (int ele = 0; ele < kWmmaTile; ++ele) {
                 b_frag[wn][ele] = Bs[buf][ele][n_local];
             }
         }
 
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
 #pragma unroll
             for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
                 c_frag[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                     a_frag[wm], b_frag[wn], c_frag[wm][wn]);
             }
         }
 
         __syncthreads();
         buf = next_buf;
     }
 
     // Store fp32 partials into workspace[k_split_id, *, *].
     float* my_ws = workspace + (size_t)k_split_id * M * N;
 #pragma unroll
     for (int wm = 0; wm < WAVE_M_TILES; ++wm) {
 #pragma unroll
         for (int wn = 0; wn < WAVE_N_TILES; ++wn) {
             int m_base = blockM + waveM0 + wm * kWmmaTile;
             int n_base = blockN + waveN0 + wn * kWmmaTile;
             int c      = n_base + laneM;
 #pragma unroll
             for (int ele = 0; ele < 8; ++ele) {
                 int r = m_base + ele * 2 + pair;
                 if (r < M && c < N) {
                     my_ws[(size_t)r * N + c] = c_frag[wm][wn][ele];
                 }
             }
         }
     }
 }
 
 // Reduce K_SPLITS partials into the final fp16 output. One thread per
 // output element; sums in fp32; downcasts on store.
 __global__ __launch_bounds__(256) void
 gemm_splitk_reduce_kernel(const float* __restrict__ workspace,
                           _Float16* __restrict__ C,
                           size_t total_elems, int k_splits)
 {
     size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= total_elems) return;
 
     float sum = 0.0f;
     for (int s = 0; s < k_splits; ++s) {
         sum += workspace[(size_t)s * total_elems + idx];
     }
     C[idx] = (_Float16)sum;
 }
 
 namespace {
     // Per-process workspace cache for split-K phase-1 partials.
     // Grows on demand, never shrinks. Single-stream-safe only -- see the
     // commentary above the phase-1 kernel.
     float* g_splitk_workspace = nullptr;
     size_t g_splitk_workspace_bytes = 0;
 }
 
 static int ensure_splitk_workspace(size_t bytes) {
     if (bytes <= g_splitk_workspace_bytes) return 0;
     if (g_splitk_workspace) {
         hipFree(g_splitk_workspace);
         g_splitk_workspace = nullptr;
         g_splitk_workspace_bytes = 0;
     }
     hipError_t err = hipMalloc(reinterpret_cast<void**>(&g_splitk_workspace),
                                bytes);
     if (err != hipSuccess) {
         g_splitk_workspace = nullptr;
         g_splitk_workspace_bytes = 0;
         return -1;
     }
     g_splitk_workspace_bytes = bytes;
     return 0;
 }
 
 template<int K_SPLITS>
 static int launch_splitk(hipStream_t stream, const _Float16* A,
                          const _Float16* B, _Float16* C,
                          int M, int K, int N) {
     constexpr int BM = 128;
     constexpr int BN = 128;
 
     size_t total_elems = (size_t)M * (size_t)N;
     size_t ws_bytes    = (size_t)K_SPLITS * total_elems * sizeof(float);
     if (ensure_splitk_workspace(ws_bytes) != 0) return -1;
 
     dim3 p1_block(256);
     dim3 p1_grid((N + BN - 1) / BN, (M + BM - 1) / BM, K_SPLITS);
     gemm_wmma_fp16_splitk_phase1_kernel<K_SPLITS>
         <<<p1_grid, p1_block, 0, stream>>>(A, B, g_splitk_workspace, M, K, N);
 
     constexpr int RED_BLOCK = 256;
     size_t red_blocks = (total_elems + RED_BLOCK - 1) / RED_BLOCK;
     dim3 p2_grid((unsigned)red_blocks);
     dim3 p2_block(RED_BLOCK);
     gemm_splitk_reduce_kernel<<<p2_grid, p2_block, 0, stream>>>(
         g_splitk_workspace, C, total_elems, K_SPLITS);
 
     return hipGetLastError() == hipSuccess ? 0 : -1;
 }
 
 extern "C"
 int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                        void* C, int M, int K, int N) {
     hipStream_t s = (hipStream_t)stream;
     const _Float16* dA = (const _Float16*)A;
     const _Float16* dB = (const _Float16*)B;
     _Float16*       dC = (_Float16*)C;
 
     // Small-M (decode) fast path: BM=128 in the WMMA kernel wastes
     // (BM-M)/BM of every fragment. Route to a coalesced-load GEMV kernel
     // when the output dimension is wide enough that GEMV's
     // bandwidth-friendly access pattern beats v3's wasted-tile compute.
     // The threshold N >= 8192 was picked from the LLM-shape sweep on
     // gfx1151: at smaller N (e.g. attn-out 1x4096x4096) v3's tile reuse
     // still wins despite the M waste; at FFN-/vocab-sized N the GEMV pulls
     // ahead (1.4ms vs 2.1ms on 1x4096x14336).
     if (M <= 4 && N >= 8192) {
         return launch_gemv(s, dA, dB, dC, M, K, N);
     }
 
     // Medium-M (batch decode / small chunked prefill) fast path: v3 uses
     // BM=128 so M in [5, 16] wastes 87-94% of every WMMA M-fragment. The
     // BM=16 kernel keeps M utilization ~100% (1 M-tile per block).
     //
     // The dispatch is intentionally narrower than the "BM<=128 wastes
     // fragments" intuition would suggest. Measured on gfx1151 (LLM
     // M=8/16/32 x K=4096 x N=4096/14336 sweep):
     //   * v3 has 8 waves/block; BM=16 only 4. For shapes where the grid
     //     is small enough that v3 stays under the CU's ~32-wave max,
     //     v3's higher waves/block wins on latency hiding even though it
     //     wastes M-fragments.
     //   * BM=16 wins only when the grid is large enough that v3
     //     oversubscribes (waves/CU > 32). Empirically that happens for
     //     M <= 16 AND N >= 8192 (e.g. 16x4096x14336: v3 2.25ms vs
     //     BM=16 1.80ms; 20% win). Outside that window BM=16 ties or
     //     regresses (16x4096x4096: v3 0.60ms vs BM=16 0.80ms).
     //   * hipBLASLt still beats both on these shapes, so this only helps
     //     when the WMMA dispatch is forced on; it does not flip the
     //     default-path decision.
     if (M >= 5 && M <= 16 && N >= 8192) {
         return launch_bm16(s, dA, dB, dC, M, K, N);
     }
 
     // Split-K dispatch is OPT-IN via HIPDNN_EP_GEMM_WMMA_SPLITK=1. The
     // kernel is implemented (see gemm_wmma_fp16_splitk_phase1_kernel and
     // gemm_splitk_reduce_kernel above) but measurements on gfx1151 showed
     // it is neutral-or-negative vs v3 across every LLM-sweep shape that
     // matched the eligibility criteria (small K, M >= 256). The cost of
     // the fp32 partial writes and the reduce pass exceeds the benefit of
     // K-parallel scheduling. Left in place behind an env gate so a future
     // session can revisit if kernel tuning closes the gap.
     static const bool splitk_enabled = [] {
         char buf[8];
         unsigned long n = GetEnvironmentVariableA(
             "HIPDNN_EP_GEMM_WMMA_SPLITK", buf, sizeof(buf));
         return n > 0 && buf[0] >= '1';
     }();
     if (splitk_enabled && M >= 256 && K <= 2880 && (K % 16) == 0) {
         int rc = launch_splitk<2>(s, dA, dB, dC, M, K, N);
         if (rc == 0) return 0;
         // fall through to v3 on workspace-alloc failure
     }
 
     // Default: v3 LDS-staged WMMA kernel.
     constexpr int BM = 128;
     constexpr int BN = 128;
     dim3 block(256);
     dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
     gemm_wmma_fp16_kernel<<<grid, block, 0, s>>>(dA, dB, dC, M, K, N);
     return hipGetLastError() == hipSuccess ? 0 : -1;
 }