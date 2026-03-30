/*
 * Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * MatMulNBits — Fused Dequant + MatMul for int4 packed weights (FP16)
 *
 * Two execution paths:
 *
 *   1. RDNA3 WMMA fast path (when M%128==0, N%128==0, K%32==0, batch==1):
 *      Uses WMMA intrinsics with double-buffered shared memory, grid swizzling,
 *      and inline-asm batched dequant. Requires gfx11xx/gfx12xx.
 *
 *      Data layout for the WMMA path:
 *        A      : FP16 col-major (M x K), stride lda = M
 *        B      : uint4 packed (N x K/2), row-major per column
 *        scales : FP16 (N x num_groups_k), row-major
 *        zeros  : FP16 (N x num_groups_k), row-major  (or NULL)
 *        output : FP16 col-major (M x N), stride ldc = M
 *
 *   2. Naive fallback (arbitrary alignment, batched, row-major):
 *        A      : FP16 row-major [batch, M, K]
 *        B      : uint4 packed [N, k_blocks, blob_size]
 *        scales : FP16 [N, k_blocks]
 *        zeros  : uint8 [N, k_blocks]  (or NULL, default zp=8)
 *        output : FP16 row-major [batch, M, N]
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

/* =======================================================================
 * Section 1: Naive fallback kernel (row-major, any alignment, batched)
 * ======================================================================= */

static constexpr int kBlockDim = 16;

__global__ void matmul_nbits_kernel(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks = (K + block_size - 1) / block_size;
  int64_t blob_size = block_size / 2;

  float acc = 0.0f;
  const __half* a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * k_blocks * blob_size;
  const __half* s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = static_cast<float>(s_row[blk]);
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + blk])
                       : 8.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end = k_start + block_size;
    if (k_end > K) {
      k_end = K;
    }
    const uint8_t* b_block = b_row + blk * blob_size;

    for (int64_t k = k_start; k < k_end; k++) {
      int64_t in_blk = k - k_start;
      int64_t byte_off = in_blk / 2;
      int nibble = static_cast<int>(in_blk & 1);
      uint8_t packed = b_block[byte_off];
      float qval = (nibble == 0) ? static_cast<float>(packed & 0xF)
                                 : static_cast<float>(packed >> 4);

      acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
    }
  }

  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}

/* =======================================================================
 * Section 2: RDNA3 WMMA Fused GEMM + Dequantization Kernel
 *
 * Computes C = A x dequant(B_packed)^T  where:
 *   A        : FP16 col-major (M x K), stride lda
 *   B_packed : uint4 packed into bytes (N x K/2), row-major per column
 *   scales   : FP16 (N x num_groups_k), per-column per-group scale
 *   zeros    : FP16 (N x num_groups_k), per-column per-group zero point
 *   C        : FP16 col-major (M x N), stride ldc
 *
 * Requires RDNA3+ (gfx11xx/gfx12xx) for WMMA intrinsics.
 * M must be multiple of 128, N must be multiple of 128, K must be multiple
 * of 32.
 * ======================================================================= */

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define WMMA_TILE 16

template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS>
__device__ __forceinline__
void GemmFp16U4Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BM_T / (WT_M * WT_N);

    constexpr int PAD_K   = 2;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int A_VL    = (BM_T * BK_T) / (THR_L * 4);
    constexpr int A_GRP_2 = BM_T / 2;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    static_assert(A_VL >= 1, "A_VL must be >= 1");
    static_assert(THR_L * 8 == BN_T * BK_T, "B loading balance: THR*8 must equal BN*BK");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw       = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
    const int super    = block_id / (sw * m_tiles);
    const int rem      = block_id % (sw * m_tiles);
    int by             = rem / sw;
    int bx             = super * sw + rem % sw;
    if(bx >= n_tiles)
    {
        bx = block_id % n_tiles;
        by = block_id / n_tiles;
    }
    const int row0 = by * BM_T;
    const int col0 = bx * BN_T;

    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;
    const int b_n_g       = col0 + b_col;
    const int b_byte_base = b_n_g * (K >> 1) + (b_k8 >> 1);
    const int b_gid_base  = b_n_g * num_groups_k;

    int      cached_k_grp = -1;
    int      next_grp_k = 0;
    _Float16 cached_s = 0;
    [[maybe_unused]] _Float16 neg_zs = 0;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for(int i = 0; i < WT_M; i++)
#pragma unroll
        for(int j = 0; j < WT_N; j++)
#pragma unroll
            for(int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    auto loadTile = [&](int buf, int t) __attribute__((always_inline))
    {
        uint32_t four = *reinterpret_cast<const uint32_t*>(
            &B_packed[b_byte_base + (t >> 1)]);

        uint32_t a_gl1[A_VL], a_gl2[A_VL];
        int      a_m2[A_VL],  a_k2[A_VL];
#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            int vid   = tid + ld * THR_L;
            a_m2[ld]  = (vid % A_GRP_2) * 2;
            a_k2[ld]  = (vid / A_GRP_2) * 2;
            a_gl1[ld] = *reinterpret_cast<const uint32_t*>(
                              &A[(t + a_k2[ld]) * lda + row0 + a_m2[ld]]);
            a_gl2[ld] = *reinterpret_cast<const uint32_t*>(
                              &A[(t + a_k2[ld] + 1) * lda + row0 + a_m2[ld]]);
        }

        if(__builtin_expect(t >= next_grp_k, 0))
        {
            cached_k_grp++;
            next_grp_k += group_size;
            cached_s   = scales[cached_k_grp + b_gid_base];
            if constexpr(USE_ZEROS)
                neg_zs = -zeros[cached_k_grp + b_gid_base] * cached_s;
        }
        _Float16 s16 = cached_s;

        _Float16 n0, n1, n2, n3, n4, n5, n6, n7;
        asm volatile(
            "v_and_b32 %0, 15, %8\n\t"
            "v_lshrrev_b32 %1, 4, %8\n\t"
            "v_lshrrev_b32 %2, 8, %8\n\t"
            "v_lshrrev_b32 %3, 12, %8\n\t"
            "v_lshrrev_b32 %4, 16, %8\n\t"
            "v_lshrrev_b32 %5, 20, %8\n\t"
            "v_lshrrev_b32 %6, 24, %8\n\t"
            "v_lshrrev_b32 %7, 28, %8\n\t"
            "v_and_b32 %1, 15, %1\n\t"
            "v_and_b32 %2, 15, %2\n\t"
            "v_and_b32 %3, 15, %3\n\t"
            "v_and_b32 %4, 15, %4\n\t"
            "v_and_b32 %5, 15, %5\n\t"
            "v_and_b32 %6, 15, %6\n\t"
            "v_cvt_f16_u16 %0, %0\n\t"
            "v_cvt_f16_u16 %1, %1\n\t"
            "v_cvt_f16_u16 %2, %2\n\t"
            "v_cvt_f16_u16 %3, %3\n\t"
            "v_cvt_f16_u16 %4, %4\n\t"
            "v_cvt_f16_u16 %5, %5\n\t"
            "v_cvt_f16_u16 %6, %6\n\t"
            "v_cvt_f16_u16 %7, %7"
            : "=&v"(n0), "=&v"(n1), "=&v"(n2), "=&v"(n3),
              "=&v"(n4), "=&v"(n5), "=&v"(n6), "=&v"(n7)
            : "v"(four)
        );
        _Float16 bv[8];
        bv[0] = n0 * s16; bv[1] = n1 * s16; bv[2] = n2 * s16; bv[3] = n3 * s16;
        bv[4] = n4 * s16; bv[5] = n5 * s16; bv[6] = n6 * s16; bv[7] = n7 * s16;
        if constexpr(USE_ZEROS) {
            _Float16 nzs = neg_zs;
#pragma unroll
            for(int i = 0; i < 8; i++)
                bv[i] += nzs;
        }
        *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = *reinterpret_cast<uint2*>(&bv[0]);
        *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = *reinterpret_cast<uint2*>(&bv[4]);

#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            uint32_t g1 = a_gl1[ld], g2 = a_gl2[ld];
            uint32_t pack_even = (g1 & 0xFFFF) | ((g2 & 0xFFFF) << 16);
            uint32_t pack_odd  = (g1 >> 16)     | (g2 & 0xFFFF0000);
            *reinterpret_cast<uint32_t*>(&smA[buf][a_m2[ld]  ][a_k2[ld]]) = pack_even;
            *reinterpret_cast<uint32_t*>(&smA[buf][a_m2[ld]+1][a_k2[ld]]) = pack_odd;
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline))
    {
#pragma unroll
        for(int ks = 0; ks < K_STEPS; ks++)
        {
            half16 b_frag[WT_N];
#pragma unroll
            for(int wn = 0; wn < WT_N; wn++)
            {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < WT_M; wm++)
            {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];

#pragma unroll
                for(int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    loadTile(0, 0);
    __syncthreads();

    int buf = 0;
    for(int t = BK_T; t < K; t += BK_T)
    {
        int nxt = 1 - buf;
        loadTile(nxt, t);
        computeWMMA(buf);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for(int wm = 0; wm < WT_M; wm++)
    {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for(int wn = 0; wn < WT_N; wn++)
        {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
            for(int e = 0; e < 8; e++)
            {
                int r = e * 2 + sub;
                __builtin_nontemporal_store((_Float16)acc[wm][wn][e],
                                            &C[(nbase + lane) * ldc + mbase + r]);
            }
        }
    }
}

/* WMMA kernel entry points with per-variant GPU attributes */

#define GEMM_WG_SZ(BM, BN, WM, WN) \
    ((BM / (WM * 16)) * (BN / (WN * 16)) * 32)

template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
__attribute__((amdgpu_num_vgpr(80)))
void MatMulNBitsWMMA_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U4Impl<BM_T, BN_T, WT_M, WT_N, false>(
        M, N, K, A, lda, B_packed, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}

template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
__attribute__((amdgpu_num_vgpr(80)))
void MatMulNBitsWMMA_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U4Impl<BM_T, BN_T, WT_M, WT_N, true>(
        M, N, K, A, lda, B_packed, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}

/* =======================================================================
 * Section 3: Post-WMMA bias addition (col-major output)
 *
 * C is col-major [M x N] with ldc = M.
 * bias is [N]. Adds bias[n] to every row of column n.
 * ======================================================================= */

__global__ void matmul_nbits_add_bias_colmajor(
    _Float16* __restrict__ C,
    const _Float16* __restrict__ bias,
    int M, int N, int ldc)
{
    int m = static_cast<int>(blockIdx.y) * 256 + threadIdx.x;
    int n = static_cast<int>(blockIdx.x);
    if (m < M && n < N)
        C[n * ldc + m] += bias[n];
}

/* =======================================================================
 * Section 4: Host-side launcher (extern "C")
 * ======================================================================= */

extern "C" int hip_matmul_nbits(
    void* stream,
    const void* A,
    const void* B,
    const void* scales,
    const void* zero_points,
    const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes) {
  if (bits != 4) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only bits=4 supported, "
            "got %lld\n",
            (long long)bits);
    return -1;
  }

  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  /* -------------------------------------------------------------------
   * RDNA3 WMMA fast path
   *
   * Conditions: fp16, batch==1, M%128==0, N%128==0, K%32==0
   *
   * Data layout requirements (different from the naive fallback):
   *   - A and output are col-major with leading dimension = M
   *   - zero_points (if not NULL) must be FP16 [N, num_groups_k]
   * ------------------------------------------------------------------- */
  bool can_use_wmma = (batch_count == 1)
                   && (M % 128 == 0)
                   && (N % 128 == 0)
                   && (K % 32 == 0);

  if (can_use_wmma) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: WMMA fast path M=%lld N=%lld "
        "K=%lld bs=%lld zp=%s bias=%s\n",
        (long long)M, (long long)N, (long long)K, (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    int iM  = static_cast<int>(M);
    int iN  = static_cast<int>(N);
    int iK  = static_cast<int>(K);
    int gs  = static_cast<int>(block_size);
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    int lda = iM;
    int ldc = iM;

    const auto* a_ptr = static_cast<const _Float16*>(A);
    const auto* b_ptr = static_cast<const unsigned char*>(B);
    const auto* s_ptr = static_cast<const _Float16*>(scales);
    const auto* z_ptr = static_cast<const _Float16*>(zero_points);
    auto*       c_ptr = static_cast<_Float16*>(output);

    constexpr int BM = 128, BN = 128;
    constexpr int WM = BM / (2 * 16);
    constexpr int WN = BN / (2 * 16);
    constexpr int THR = WM * WN * 32;

    dim3 grid(static_cast<unsigned>(iN / BN),
              static_cast<unsigned>(iM / BM));
    dim3 block(THR);

    if (z_ptr) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(MatMulNBitsWMMA_ZP<BM, BN, 2, 2, 8>),
          grid, block, 0, hip_stream,
          iM, iN, iK, a_ptr, lda, b_ptr, s_ptr, z_ptr,
          gs, ngk, c_ptr, ldc, 4);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(MatMulNBitsWMMA_NoZP<BM, BN, 2, 2, 8>),
          grid, block, 0, hip_stream,
          iM, iN, iK, a_ptr, lda, b_ptr, s_ptr, z_ptr,
          gs, ngk, c_ptr, ldc, 4);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits WMMA launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }

    if (bias) {
      dim3 bias_grid(static_cast<unsigned>(iN),
                     static_cast<unsigned>((iM + 255) / 256));
      dim3 bias_block(256);
      hipLaunchKernelGGL(
          matmul_nbits_add_bias_colmajor, bias_grid, bias_block,
          0, hip_stream,
          c_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc);

      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_matmul_nbits bias launch failed: %s\n",
                hipGetErrorString(err));
      }
    }

    return static_cast<int>(err);
  }

  /* -------------------------------------------------------------------
   * Naive fallback (row-major, batched, uint8 zero points)
   * ------------------------------------------------------------------- */
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_matmul_nbits: naive fallback M=%lld N=%lld K=%lld "
      "batch=%lld bits=%lld bs=%lld elem=%lld zp=%s bias=%s\n",
      (long long)M, (long long)N, (long long)K, (long long)batch_count,
      (long long)bits, (long long)block_size, (long long)element_size_bytes,
      zero_points ? "yes" : "null", bias ? "yes" : "null");

  dim3 blk(kBlockDim, kBlockDim);
  dim3 grd(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
           static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
           static_cast<unsigned>(batch_count));

  hipLaunchKernelGGL(
      matmul_nbits_kernel, grd, blk, 0, hip_stream,
      static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
      static_cast<const __half*>(scales),
      static_cast<const uint8_t*>(zero_points),
      static_cast<const __half*>(bias), static_cast<__half*>(output),
      M, N, K, block_size);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits naive launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
