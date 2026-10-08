/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Internal packed-U4 / FP16 WMMA tile core shared by dense MatMulNBits and
 * descriptor-driven grouped/ragged kernels. This header contains device-only
 * implementation; it does not define a public ABI or launch policy.
 */

#ifndef HIPDNN_MATMUL_NBITS_WMMA_CORE_H
#define HIPDNN_MATMUL_NBITS_WMMA_CORE_H

#include "hip_arch_compat.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// half16 holds one WMMA A/B fragment; its vector width matches
// HIPDNN_WMMA_FRAG_ELEMS (16 on gfx11, 8 on the gfx12-style encoding
// gfx1170/gfx12xx use -- see hip_arch_compat.h). The name is kept for
// continuity even though the width is arch-dependent.
typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float    float8 __attribute__((ext_vector_type(8)));

// Explicit 8- and 16-byte vectors for the shared-memory staging. Writing the
// loads and stores through these instead of a loop over uint32_t is what makes
// the compiler emit global_load_dwordx4 / ds_read_b128 / ds_write_b128 rather
// than a sequence of narrower accesses.
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 half2v __attribute__((ext_vector_type(2)));

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
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                                  \
  hipdnn_wmma_f32_16x16x16_f16_unavailable((a), (b), (c))
#endif

#define WMMA_TILE 16

/* LDS-only workgroup barrier.
 *
 * __syncthreads() is a full workgroup fence: on RDNA it emits s_waitcnt for both
 * counters, s_barrier, AND buffer_gl0_inv, because a workgroup can span the two
 * CUs of a WGP and global-memory coherence between them requires invalidating
 * the per-CU L0. The GEMM K-loop only ever synchronizes shared memory -- the
 * global loads are private to the thread that issues them -- so the L0 flush is
 * pure overhead, once per K-step, and it also throws away any L0 hits the next
 * tile's loads might have had.
 *
 * The "memory" clobber is what keeps the compiler from moving LDS accesses
 * across the barrier; without it the raw s_barrier intrinsic is IntrNoMem and
 * reordering is legal.
 *
 * The standalone s_barrier instruction is provided by the gfx9, gfx10 and
 * gfx11 families. gfx12 replaced it with the s_barrier_signal/s_barrier_wait
 * pair, and later architectures likewise lack it, so the inline asm fails to
 * assemble there. Emit the asm only on the families that provide it and let
 * every other target -- current or future -- fall back to __syncthreads(),
 * which is always correct (a strictly stronger full-workgroup fence); it only
 * reintroduces the redundant per-K-step L0 flush the raw asm skips. */
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) &&                   \
    (defined(__GFX9__) || defined(__GFX10__) || defined(__GFX11__))
#define HIPDNN_LDS_BARRIER()                                                    \
    __asm__ __volatile__("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#else
#define HIPDNN_LDS_BARRIER() __syncthreads()
#endif

/* BK_IN decouples the K-step from the warp-tile shape.
 *
 * BK_T used to be forced to BM_T/(WT_M*WT_N), which is the value that makes each
 * thread stage exactly 8 B-nibbles per K-step. That coupling is what kept WT_N
 * at 2: a wider warp tile shrank BK_T below the 16 a WMMA step needs. Letting a
 * thread stage 16 or 32 nibbles instead frees the two to be chosen
 * independently, and WT_N=4 is worth having -- a K-step then issues WT_M*WT_N=8
 * WMMA per (WT_M+WT_N)=6 fragment loads instead of 4 per 4, so the LDS reads per
 * Matrix Core op drop by a quarter. */
template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false, bool FUSED_DQ = true,
          int BK_IN = BM_T / (WT_M * WT_N), bool ZEROS_U8 = false,
          bool ADD_BIAS = false, bool EXPLICIT_TILE = false,
          bool CHECK_K_TAIL = false, bool RUNTIME_OPTIONAL = false>
__device__ __forceinline__
void GemmFp16U4Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const void* __restrict__ B_data,
    const _Float16* __restrict__ scales,
    const void* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4,
    /* Bytes between B rows. 0 derives the arrival stride,
     * num_groups_k * (group_size/2); a larger value reads a buffer whose rows
     * were padded to break cache-set aliasing.
     *
     * Runtime rather than a template parameter so one set of instantiations
     * serves both layouts and the autotuner stays free to pick any tile for a
     * padded buffer. As a template parameter it would multiply the
     * instantiation count of every tile shape. */
    int b_row_bytes = 0,
    /* Descriptor-driven callers supply an explicit output tile and optional
     * gather map. Dense MatMulNBits leaves EXPLICIT_TILE=false, so its original
     * blockIdx/swizzle mapping is compiled unchanged. */
    int tile_row0 = 0, int tile_col0 = 0, int tile_rows = BM_T,
    const int32_t* __restrict__ a_row_ids = nullptr,
    const _Float16* __restrict__ output_bias = nullptr)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BK_IN;

    /* The LDS row stride is padded to a multiple of 8 halfs (16 B), not to the
     * 2 halfs that would be enough to break bank conflicts.
     *
     * With a 4-byte-aligned stride the WMMA fragment reads are not provably
     * 16-byte aligned, so the compiler cannot use ds_read_b128 and falls back to
     * two ds_load_2addr_b32 per 8 dwords. That doubles the LDS instruction count
     * and, because the b64 form only carries 8-bit element offsets, forces a
     * v_add_nc_u32 per address once the tile is bigger than ~1 KB: 32 LDS reads
     * plus 22 address adds per K-step in the 128x256 tile, against 16 reads and
     * no adds after this change (the b128 form has a 16-bit byte offset).
     *
     * BK_T+8 stays conflict-free for every BK_T the config table uses: the dword
     * row stride is then 12, 20 or 36, and 8 consecutive lanes reading 4 dwords
     * each cover 32 distinct banks in all three cases. */
    constexpr int PAD_K   = 8;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    /* A staging: one thread moves A_CH *contiguous* halfs of a single row, so
     * the tile costs one global load and one LDS store per chunk. The previous
     * scheme moved two halfs from each of two rows, which is the same bytes in
     * four instructions instead of one. */
    constexpr int A_HPT = (BM_T * BK_T) / THR_L;
    constexpr int A_CH  = (A_HPT >= 8) ? 8 : 4;
    constexpr int A_NCH = A_HPT / A_CH;
    constexpr int A_ROW_SPLIT = BK_T / A_CH;

    /* B staging: B_NPT nibbles (B_U32 packed words) per thread, all contiguous
     * in k, so B_TPR threads cover one B row. */
    constexpr int B_NPT = (BN_T * BK_T) / THR_L;
    constexpr int B_U32 = B_NPT / 8;
    constexpr int B_TPR = BK_T / B_NPT;

    static_assert(A_NCH >= 1, "A_NCH must be >= 1");
    static_assert(A_HPT % A_CH == 0, "A_HPT must be a multiple of A_CH");
    static_assert(BK_T % A_CH == 0, "BK_T must be a multiple of A_CH");
    static_assert(BK_T % WMMA_TILE == 0, "BK_T must be a multiple of 16");
    static_assert((K_STR * 2) % 16 == 0, "LDS row stride must be 16B aligned");
    static_assert(B_NPT >= 8 && B_NPT % 8 == 0, "B_NPT must be a multiple of 8");
    static_assert(B_U32 <= 4, "B_NPT must not exceed 32 nibbles per thread");
    static_assert(B_TPR * B_NPT == BK_T, "B_TPR must tile BK_T exactly");
    static_assert(THR_L == BN_T * B_TPR, "B staging must use every thread once");

    __shared__ __align__(16) _Float16 smA[2][BM_T][K_STR];
    __shared__ __align__(16) _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    // Block-to-tile mapping with column swizzle for L2 locality.
    // Splits the grid into a "main" region of (n_tiles/sw)*sw cols where full
    // sw-wide groups apply, and a "tail" region of (n_tiles%sw) leftover cols
    // that uses plain row-major. The previous implementation fell back to
    // row-major only when bx >= n_tiles, which produced both collisions AND
    // missing tiles when n_tiles % sw != 0, leaving those tiles with stale
    // output data from prior kernel launches.
    int row0;
    int col0;
    if constexpr (EXPLICIT_TILE) {
        row0 = tile_row0;
        col0 = tile_col0;
    } else {
        const int n_tiles  = gridDim.x;
        const int m_tiles  = gridDim.y;
        int by, bx;
        if (swizzle_n <= 1) {
            // Direct mapping is used by ragged grouped WMMA: its launcher can
            // then reject blockIdx.y tiles beyond each expert's row count before
            // entering the expensive K loop.
            by = static_cast<int>(blockIdx.y);
            bx = static_cast<int>(blockIdx.x);
        } else {
            const int block_id = blockIdx.y * n_tiles + blockIdx.x;
            const int sw          = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
            const int main_cols   = (n_tiles / sw) * sw;
            const int main_blocks = main_cols * m_tiles;
            if (block_id < main_blocks) {
                const int super = block_id / (sw * m_tiles);
                const int rem   = block_id % (sw * m_tiles);
                by = rem / sw;
                bx = super * sw + rem % sw;
            } else {
                const int tail_id   = block_id - main_blocks;
                const int tail_cols = n_tiles - main_cols;
                by = tail_id / tail_cols;
                bx = main_cols + (tail_id % tail_cols);
            }
        }
        row0 = by * BM_T;
        col0 = bx * BN_T;
    }

    constexpr int B_COL_SHIFT = __builtin_ctz(B_TPR);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_k8        = (tid & (B_TPR - 1)) * B_NPT;
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    /* Running source pointers.
     *
     * The K-loop used to rebuild every address from (row, t, k), which the
     * compiler turns into a sign-extend / shift / 64-bit add chain: ~10 VALU ops
     * per K-step for A alone. Advancing a pointer by one K-step is two.
     *
     * B_packed rows are padded to num_groups_k * (group_size/2) bytes (ONNX
     * MatMulNBits blob layout -- the last group is padded to a full group_size
     * even when K % group_size != 0), NOT a plain K/2 bytes/row; see the
     * row-stride comment on matmul_nbits_gemv_kernel's b_base_arr. Within a row
     * the nibbles are still packed compactly from k=0 (only the tail beyond K is
     * padding), so the intra-row byte offset (b_k8 >> 1) and the per-K-step
     * advance below are unaffected -- only the per-row stride changes.  */
    [[maybe_unused]] const unsigned char* b_src_u4 = nullptr;
    [[maybe_unused]] const _Float16*      b_src_fp = nullptr;
    [[maybe_unused]] const _Float16*      s_src    = nullptr;
    [[maybe_unused]] const _Float16*      z_src_f16 = nullptr;
    [[maybe_unused]] const uint8_t*       z_src_u8  = nullptr;
    if constexpr (FUSED_DQ) {
        // 0 means the arrival layout, whose stride carries the ONNX group
        // padding described above; a repacked buffer passes its own.
        const int row_bytes =
            b_row_bytes ? b_row_bytes : (num_groups_k * (group_size >> 1));
        b_src_u4 = static_cast<const unsigned char*>(B_data) +
                   (static_cast<size_t>(b_n_g) * row_bytes + (b_k8 >> 1));
        s_src = scales + static_cast<size_t>(b_n_g) * num_groups_k;
        if constexpr (USE_ZEROS) {
            if constexpr (ZEROS_U8) {
                if constexpr (RUNTIME_OPTIONAL) {
                    if (zeros)
                        z_src_u8 = static_cast<const uint8_t*>(zeros) +
                                   static_cast<size_t>(b_n_g) * num_groups_k;
                } else {
                    z_src_u8 = static_cast<const uint8_t*>(zeros) +
                               static_cast<size_t>(b_n_g) * num_groups_k;
                }
            } else {
                if constexpr (RUNTIME_OPTIONAL) {
                    if (zeros)
                        z_src_f16 = static_cast<const _Float16*>(zeros) +
                                    static_cast<size_t>(b_n_g) * num_groups_k;
                } else {
                    z_src_f16 = static_cast<const _Float16*>(zeros) +
                                static_cast<size_t>(b_n_g) * num_groups_k;
                }
            }
        }
    } else {
        b_src_fp = static_cast<const _Float16*>(B_data) +
                   (static_cast<size_t>(b_n_g) * K + b_k8);
    }

    int              a_row[A_NCH];
    int              a_koff[A_NCH];
    bool             a_valid[A_NCH];
    const _Float16*  a_src[A_NCH];
#pragma unroll
    for (int c = 0; c < A_NCH; c++) {
        const int idx = tid + c * THR_L;
        a_row[c]      = idx / A_ROW_SPLIT;
        a_koff[c]     = (idx % A_ROW_SPLIT) * A_CH;
        const int logical_row = row0 + a_row[c];
        a_valid[c] = logical_row < M;
        if constexpr (EXPLICIT_TILE)
            a_valid[c] = a_valid[c] && a_row[c] < tile_rows;
        int source_row = a_valid[c] ? logical_row : 0;
        if (a_valid[c] && a_row_ids)
            source_row = a_row_ids[logical_row];
        a_src[c] = A + static_cast<size_t>(source_row) * lda + a_koff[c];
    }

    float8 acc[WT_M][WT_N];
#pragma unroll
    for(int i = 0; i < WT_M; i++)
#pragma unroll
        for(int j = 0; j < WT_N; j++)
#pragma unroll
            for(int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    /* --- Register-prefetch pipeline: split load from dequant+store ---
     *
     * issueLoads(t): fire off global memory reads into registers (non-blocking).
     * storeToShared(buf): dequant B in registers, write A & B to smem.
     *
     * The K-loop interleaves: issueLoads → computeWMMA → storeToShared → sync
     * so that global loads overlap with Matrix Core WMMA execution.
     */

    uint32_t pf_a[A_NCH][A_CH / 2];
    [[maybe_unused]] uint32_t pf_b4[B_U32] = {};
    [[maybe_unused]] u32x4    pf_bfp[B_U32] = {};

    /* Quant-group state.
     *
     * grp_k_end is the first k beyond the group whose scale/zero is currently
     * held in s_pk / bias_pk. k advances monotonically, so the group index is
     * carried across K-steps rather than recomputed: `(t + b_k8) / group_size`
     * with a runtime group_size expands into a ~13-instruction division
     * sequence on every K-step.
     *
     * s_pk and bias_pk hold the group constants pre-broadcast into both halves
     * of a 32-bit register so the dequant can use v_pk_* packed math directly. */
    [[maybe_unused]] int      grp_cur   = 0;
    [[maybe_unused]] int      grp_k_end = 0;
    [[maybe_unused]] uint32_t s_pk      = 0;
    [[maybe_unused]] uint32_t bias_pk   = 0;
    [[maybe_unused]] uint32_t zp_u4     = 8;

    // fp16 1024.0. OR-ing a 4-bit value into the mantissa of 1024.0 yields
    // exactly 1024+q, which is the integer-to-fp16 conversion without any
    // v_cvt: see the dequant in storeToShared.
    constexpr uint32_t kMagicPk = 0x64006400u;

    auto loadGroupConstants = [&](int grp) __attribute__((always_inline))
    {
        const _Float16 s16 = s_src[grp];
        // Subtracting (1024 + zp) from (1024 + q) is exact in fp16 -- both
        // operands and the result are exactly representable -- so the magic-bias
        // conversion loses no accuracy. Folding it into the scale instead
        // (fma(1024+q, s, -(1024+zp)*s)) would not: the fp16 constant
        // -(1024+zp)*s carries a rounding error of half an ULP at magnitude
        // ~1024*s, which is a few percent of the (q-zp)*s result.
        _Float16 zp = (_Float16)8;
        if constexpr (USE_ZEROS) {
            if constexpr (ZEROS_U8) {
                if constexpr (RUNTIME_OPTIONAL) {
                    if (z_src_u8) {
                        zp_u4 = z_src_u8[grp];
                        zp = static_cast<_Float16>(zp_u4);
                    }
                } else {
                    zp_u4 = z_src_u8[grp];
                    zp = static_cast<_Float16>(zp_u4);
                }
            } else {
                if constexpr (RUNTIME_OPTIONAL) {
                    if (z_src_f16) zp = z_src_f16[grp];
                } else {
                    zp = z_src_f16[grp];
                }
            }
        }
        const _Float16 bias = -((_Float16)1024 + zp);
        const uint16_t s_bits = __builtin_bit_cast(uint16_t, s16);
        const uint16_t b_bits = __builtin_bit_cast(uint16_t, bias);
        s_pk    = (uint32_t(s_bits) << 16) | s_bits;
        bias_pk = (uint32_t(b_bits) << 16) | b_bits;
    };

    if constexpr (FUSED_DQ) {
        if (b_valid) {
            grp_cur   = b_k8 / group_size;
            grp_k_end = (grp_cur + 1) * group_size;
            loadGroupConstants(grp_cur);
        }
    }

    auto issueLoads = [&](int k_abs) __attribute__((always_inline))
    {
#pragma unroll
        for(int c = 0; c < A_NCH; c++)
        {
            const bool ok = BOUNDS_CHECK ? a_valid[c] : true;
            const int tile_k = k_abs - b_k8;
            const int a_k = tile_k + a_koff[c];
            const bool full_k = CHECK_K_TAIL ? (a_k + A_CH <= K) : true;
            if (ok && full_k) {
                if constexpr (A_CH == 8)
                    *reinterpret_cast<u32x4*>(pf_a[c]) =
                        *reinterpret_cast<const u32x4*>(a_src[c]);
                else
                    *reinterpret_cast<u32x2*>(pf_a[c]) =
                        *reinterpret_cast<const u32x2*>(a_src[c]);
            } else {
#pragma unroll
                for (int e = 0; e < A_CH / 2; e++) pf_a[c][e] = 0;
                if constexpr (CHECK_K_TAIL) {
                    if (ok && a_k < K) {
                        _Float16* tail = reinterpret_cast<_Float16*>(pf_a[c]);
#pragma unroll
                        for (int e = 0; e < A_CH; e++)
                            if (a_k + e < K) tail[e] = a_src[c][e];
                    }
                }
            }
            a_src[c] += BK_T;
        }

        if (b_valid) {
            if constexpr (FUSED_DQ) {
                if constexpr (B_U32 == 1)
                    pf_b4[0] = *reinterpret_cast<const uint32_t*>(b_src_u4);
                else if constexpr (B_U32 == 2)
                    *reinterpret_cast<u32x2*>(pf_b4) =
                        *reinterpret_cast<const u32x2*>(b_src_u4);
                else
                    *reinterpret_cast<u32x4*>(pf_b4) =
                        *reinterpret_cast<const u32x4*>(b_src_u4);
                b_src_u4 += BK_T >> 1;
                // This thread's B_NPT nibbles are contiguous in k starting at a
                // multiple of B_NPT, and group_size is a power of two >= 16 >=
                // B_NPT for every config that sets B_NPT > 8, so they share one
                // quant group. The loop form (rather than a single compare)
                // covers BK_T > group_size, where one K-step spans several
                // groups: the 128x32 WT2x1 tile has BK_T=64 and would otherwise
                // decode the upper half with a stale scale.
                if (__builtin_expect(k_abs >= grp_k_end, 0)) {
                    do {
                        grp_cur++;
                        grp_k_end += group_size;
                    } while (k_abs >= grp_k_end);
                    loadGroupConstants(grp_cur);
                }
                if constexpr (CHECK_K_TAIL) {
                    if (__builtin_expect(k_abs + B_NPT > K, 0)) {
                        const uint32_t zp_word = zp_u4 * 0x11111111u;
#pragma unroll
                        for (int w = 0; w < B_U32; w++) {
                            int valid = K - (k_abs + w * 8);
                            valid = valid < 0 ? 0 : (valid > 8 ? 8 : valid);
                            const uint32_t keep =
                                valid == 8 ? 0xffffffffu :
                                (valid == 0 ? 0u : ((1u << (valid * 4)) - 1u));
                            pf_b4[w] = (pf_b4[w] & keep) |
                                        (zp_word & ~keep);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int w = 0; w < B_U32; w++)
                    pf_bfp[w] = reinterpret_cast<const u32x4*>(b_src_fp)[w];
                b_src_fp += BK_T;
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline))
    {
        if (b_valid) {
            if constexpr (FUSED_DQ) {
                /* uint4 -> fp16 without a single v_cvt.
                 *
                 * OR-ing a nibble into the mantissa of fp16 1024.0 gives exactly
                 * 1024+q, and v_perm_b32 places two nibbles into the low bytes
                 * of the two 16-bit lanes in one instruction. Per 8 weights that
                 * is 3 mask ops + 4 perms + 4 ORs + 4 pk_add + 4 pk_mul, against
                 * 8 extracts + 8 v_cvt_f32_ubyte0 + 8 v_cvt_f16_f32 + 4 packs +
                 * 4 pk_fma for the arithmetic conversion the compiler generates
                 * from static_cast<_Float16>(unsigned). */
                const half2v bias2 = __builtin_bit_cast(half2v, bias_pk);
                const half2v scale2 = __builtin_bit_cast(half2v, s_pk);
#pragma unroll
                for (int w = 0; w < B_U32; w++) {
                    const uint32_t packed = pf_b4[w];
                    const uint32_t lo = packed & 0x0F0F0F0Fu;         // q0 q2 q4 q6
                    const uint32_t hi = (packed >> 4) & 0x0F0F0F0Fu;  // q1 q3 q5 q7
                    u32x4 raw;
                    // Selector byte i picks byte i of {hi, lo} for output byte i:
                    // 0..3 are lo's bytes, 4..7 are hi's, 0x0C is a literal zero.
                    raw[0] = __builtin_amdgcn_perm(hi, lo, 0x0C040C00u);
                    raw[1] = __builtin_amdgcn_perm(hi, lo, 0x0C050C01u);
                    raw[2] = __builtin_amdgcn_perm(hi, lo, 0x0C060C02u);
                    raw[3] = __builtin_amdgcn_perm(hi, lo, 0x0C070C03u);
                    u32x4 out;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const half2v q =
                            __builtin_bit_cast(half2v, raw[i] | kMagicPk);
                        out[i] = __builtin_bit_cast(uint32_t, (q + bias2) * scale2);
                    }
                    reinterpret_cast<u32x4*>(&smB[buf][b_col][b_k8])[w] = out;
                }
            } else {
#pragma unroll
                for (int w = 0; w < B_U32; w++)
                    reinterpret_cast<u32x4*>(&smB[buf][b_col][b_k8])[w] = pf_bfp[w];
            }
        } else {
            const u32x4 zero4 = {0, 0, 0, 0};
#pragma unroll
            for (int w = 0; w < B_U32; w++)
                reinterpret_cast<u32x4*>(&smB[buf][b_col][b_k8])[w] = zero4;
        }

#pragma unroll
        for(int c = 0; c < A_NCH; c++)
        {
            _Float16* dst = &smA[buf][a_row[c]][a_koff[c]];
            if constexpr (A_CH == 8)
                *reinterpret_cast<u32x4*>(dst) = *reinterpret_cast<const u32x4*>(pf_a[c]);
            else
                *reinterpret_cast<u32x2*>(dst) = *reinterpret_cast<const u32x2*>(pf_a[c]);
        }
    };

    // ks_begin/ks_end let the caller run the K-step's WMMA in two pieces and put
    // the shared-memory store for the *next* tile between them.
    auto computeWMMA = [&](int buf, int ks_begin, int ks_end)
        __attribute__((always_inline))
    {
#pragma unroll
        for(int ks = ks_begin; ks < ks_end; ks++)
        {
            // 16-byte vector reads: a half16 fragment is two ds_read_b128, which
            // the padded row stride above makes legal.
            half16 b_frag[WT_N];
            const int k_off = hipdnn_wmma_k_off(sub);
#pragma unroll
            for(int wn = 0; wn < WT_N; wn++)
            {
                int noff            = (wcol * WT_N + wn) * WMMA_TILE;
                u32x4* dst          = reinterpret_cast<u32x4*>(&b_frag[wn]);
                const u32x4* src    = reinterpret_cast<const u32x4*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for(int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < WT_M; wm++)
            {
                int moff            = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                u32x4* dst          = reinterpret_cast<u32x4*>(&a_frag);
                const u32x4* src    = reinterpret_cast<const u32x4*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for(int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; i++)
                    dst[i] = src[i];

#pragma unroll
                for(int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    // issueLoads takes this thread's absolute k, which is what the quant-group
    // tracking keys on; the source pointers advance themselves.
    issueLoads(b_k8);
    storeToShared(0);
    HIPDNN_LDS_BARRIER();

    /* The store stays after the whole WMMA sequence.
     *
     * Splitting the sequence and dequantizing the next tile in the middle looks
     * like it should hide the dequant behind Matrix Core work, and it was tried:
     * it costs 3-38% depending on the tile, because the WT_N fragments have to
     * stay live across the store and the extra pressure spills into worse
     * scheduling. The compiler already interleaves the tail of the WMMA sequence
     * with the dequant on its own. */
    int buf = 0;
    for(int t = BK_T; t < K; t += BK_T)
    {
        int nxt = 1 - buf;
        issueLoads(t + b_k8);
        computeWMMA(buf, 0, K_STEPS);
        storeToShared(nxt);
        HIPDNN_LDS_BARRIER();
        buf = nxt;
    }
    computeWMMA(buf, 0, K_STEPS);

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
                int r = hipdnn_wmma_acc_row(sub, e);
                const int tile_r =
                    (wrow * WT_M + wm) * WMMA_TILE + r;
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N &&
                        (!EXPLICIT_TILE || tile_r < tile_rows)) {
                        float value = acc[wm][wn][e];
                        if constexpr (ADD_BIAS) {
                            if constexpr (RUNTIME_OPTIONAL) {
                                if (output_bias)
                                    value += static_cast<float>(output_bias[nbase + lane]);
                            } else {
                                value += static_cast<float>(output_bias[nbase + lane]);
                            }
                        }
                        C[(mbase + r) * ldc + nbase + lane] = (_Float16)value;
                    }
                } else {
                    float value = acc[wm][wn][e];
                    if constexpr (ADD_BIAS) {
                        if constexpr (RUNTIME_OPTIONAL) {
                            if (output_bias)
                                value += static_cast<float>(output_bias[nbase + lane]);
                        } else {
                            value += static_cast<float>(output_bias[nbase + lane]);
                        }
                    }
                    C[(mbase + r) * ldc + nbase + lane] = (_Float16)value;
                }
            }
        }
    }
}


#endif // HIPDNN_MATMUL_NBITS_WMMA_CORE_H
