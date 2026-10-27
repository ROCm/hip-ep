/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Extended packed-U4 / FP16 WMMA tile core, GemmFp16U4ImplX. Same tile model as
 * GemmFp16U4Impl (matmul_nbits_wmma_core.h, which every existing kWmmaConfigs
 * entry keeps using) plus these pipeline options, specialized to the one
 * contract its callers use (fused int4 dequant, fp16 zero points, dense tile
 * mapping, whole K-steps, no bias):
 *   PGR_P   global prefetch depth (1, or 2 register sets)
 *   ADDR_P  1 = wave-uniform source base plus 32-bit per-thread offsets
 *   PIPE_P  1 = store-first: write tile i+1 to LDS, issue tile i+2, then run
 *           tile i's WMMAs (one register set)
 *   SCHED_P sched_group_barrier pattern on the store-first step: 1 full,
 *           2 LDS reads + global loads, 3 dequant VALU only
 *   EPI_P   1 = stage C through LDS and write 16-byte rows
 *   XOR_P   1 = XOR-swizzle 16-byte LDS chunks instead of padding rows
 * Device-only implementation; no public ABI.
 */

#ifndef HIPDNN_MATMUL_NBITS_WMMA_CORE_X_H
#define HIPDNN_MATMUL_NBITS_WMMA_CORE_X_H

#include "matmul_nbits_wmma_core.h"

#include <type_traits>

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
          bool BOUNDS_CHECK, int BK_IN, int PGR_P = 1, int ADDR_P = 0,
          int PIPE_P = 0, int SCHED_P = 0, int EPI_P = 0, int XOR_P = 0>
__device__ __forceinline__
void GemmFp16U4ImplX(
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
    int b_row_bytes = 0)
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
    constexpr int PAD_K   = XOR_P ? 0 : 8;
    constexpr int K_STR   = BK_T + PAD_K;
    // XOR swizzle: 16-byte chunk c of row r is stored at chunk c ^ ((r / XP) % XC),
    // XC chunks per row and XP rows per 128 bytes, so 8 consecutive rows read
    // at one chunk hit 8 different 16-byte bank groups.
    constexpr int XC = BK_T / 8;
    constexpr int XP = (BK_T * 2 >= 128) ? 1 : 128 / (BK_T * 2);
    static_assert(!XOR_P || (XC & (XC - 1)) == 0, "XOR swizzle needs a power-of-two chunk count");
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    /* A staging: one thread moves A_CH *contiguous* halfs of a single row, so
     * the tile costs one global load and one LDS store per chunk. The previous
     * scheme moved two halfs from each of two rows, which is the same bytes in
     * four instructions instead of one. */
    constexpr int A_HPT = (BM_T * BK_T) / THR_L;
    constexpr int A_CH  = (A_HPT >= 8) ? 8 : 4;
    constexpr int A_NCH = A_HPT / A_CH;
    constexpr int A_ROW_SPLIT = BK_T / A_CH;

    /* B staging: a thread moves B_NCH chunks of B_CW elements, each contiguous
     * in k. B_TPR threads cover one row's K-step and a pass of all threads
     * covers THR/B_TPR rows, so all of a thread's chunks sit at the same k
     * offset, B_ROWS rows apart. Every chunk therefore shares one quant group
     * index, and only its row (scale, zero point, validity) differs.
     *
     * With one chunk per thread (B_NCH = 1) this is the original mapping, which
     * tied the thread count to BN; several chunks lift that, e.g. a 96-wide B
     * tile with 128 threads and BK = 64 is 3 chunks of 16.
     *
     * The chunk is the whole per-thread share when it is <= 32 and tiles BK_T
     * in a power-of-two number of chunks (the original mapping), else the
     * widest of 32/16/8 dividing both the share and BK_T. */
    constexpr int B_EPT = (BN_T * BK_T) / THR_L;
    constexpr bool B_WHOLE = B_EPT <= 32 && BK_T % B_EPT == 0 &&
                             (((BK_T / B_EPT) & (BK_T / B_EPT - 1)) == 0);
    constexpr int B_CW  = B_WHOLE ? B_EPT
                        : (B_EPT % 32 == 0 && BK_T % 32 == 0) ? 32
                        : (B_EPT % 16 == 0 && BK_T % 16 == 0) ? 16 : 8;
    constexpr int B_NCH  = B_EPT / B_CW;
    constexpr int B_U32  = B_CW / 8;
    constexpr int B_TPR  = BK_T / B_CW;
    constexpr int B_ROWS = THR_L / B_TPR;

    /* Global prefetch depth. 1 keeps one K-step of A/B in registers while the
     * current one runs on the Matrix Cores; 2 keeps two (hipBLASLt's PGR2), so
     * a load has two compute phases to land. With depth 2 the quant-group
     * constants travel with their tile (one scale/zero load per chunk per
     * K-step, packed when the tile is dequantized), because the running
     * constants would already belong to the tile after next. */
    constexpr int PGR = PGR_P;
    /* Pipeline and LDS knobs (all 0 = the original structure).
     *   PIPE_P=1   store-first: each step writes tile i+1 (already in
     *              registers) to the other LDS buffer, issues the global
     *              loads for tile i+2 and then runs tile i's WMMAs. One
     *              register set, global reads two steps ahead of their WMMAs,
     *              and tile i+1's dequant is independent of tile i's WMMAs.
     *   SCHED_P    sched_group_barrier pattern over the store-first step
     *              (needs PIPE_P=1): 1 interleaves WMMA with LDS reads/writes,
     *              global loads and dequant VALU, 2 with LDS reads and global
     *              loads only, 3 with dequant VALU only.
     *   EPI_P=1    stage C through LDS and write it as 16-byte rows.
     *   XOR_P=1    XOR-swizzle 16-byte LDS chunks by row instead of padding
     *              rows (PAD_K is then 0). */
    constexpr bool PIPE_SF = PIPE_P != 0;
    static_assert(!PIPE_SF || PGR == 1, "store-first uses one register set");
    static_assert(!SCHED_P || PIPE_SF, "the schedule pattern is for the store-first step");

    static_assert(A_NCH >= 1, "A_NCH must be >= 1");
    static_assert(A_HPT % A_CH == 0, "A_HPT must be a multiple of A_CH");
    static_assert(BK_T % A_CH == 0, "BK_T must be a multiple of A_CH");
    static_assert(BK_T % WMMA_TILE == 0, "BK_T must be a multiple of 16");
    static_assert((K_STR * 2) % 16 == 0, "LDS row stride must be 16B aligned");
    static_assert(B_EPT * THR_L == BN_T * BK_T, "B staging must divide the threads");
    static_assert(B_CW >= 8 && B_CW % 8 == 0 && B_CW <= 32, "B chunk must be 8, 16, 24 or 32");
    static_assert(B_NCH * B_CW == B_EPT, "B chunks must tile the per-thread share");
    static_assert(B_TPR * B_CW == BK_T, "B chunks must tile BK_T");
    static_assert((B_TPR & (B_TPR - 1)) == 0, "threads per B row must be a power of two");
    static_assert(THR_L % B_TPR == 0, "a pass must cover whole B rows");
    static_assert(B_NCH * B_ROWS == BN_T, "B passes must cover BN rows");
    static_assert(PGR == 1 || PGR == 2, "prefetch depth 1 or 2");

    /* Source addressing. 0: one 64-bit pointer per A and B chunk, each
     * advanced per K-step (2 VGPRs and 2 VALU ops per chunk). 1: the tile's
     * base is wave-uniform (SGPRs, advanced with scalar ops) and each chunk is
     * that base plus one per-thread 32-bit offset plus a uniform chunk step, so
     * the loads can use the scalar-base global form. */
    constexpr bool UADDR = ADDR_P != 0;
    static_assert(!UADDR || THR_L % A_ROW_SPLIT == 0, "A chunks must share a k offset");

    // One raw buffer holds the A/B tiles during the K-loop and the C tile
    // afterwards when EPI_P; otherwise the two arrays below are used, as in the
    // tuned kernels.
    constexpr int C_STR = (sizeof(_Float16) * BM_T * (BN_T + 8) <= 65536) ? BN_T + 8 : BN_T;
    constexpr size_t SMA_BYTES = sizeof(_Float16) * 2 * BM_T * K_STR;
    constexpr size_t SMB_BYTES = sizeof(_Float16) * 2 * BN_T * K_STR;
    constexpr size_t SMC_BYTES = sizeof(_Float16) * BM_T * C_STR;
    constexpr size_t SM_BYTES = (SMA_BYTES + SMB_BYTES) > SMC_BYTES ? (SMA_BYTES + SMB_BYTES) : SMC_BYTES;
    static_assert(!EPI_P || SM_BYTES <= 65536, "staged epilogue does not fit in LDS");
    __shared__ __align__(16) unsigned char sm_raw[EPI_P ? SM_BYTES : 16];
    __shared__ __align__(16) _Float16 smA_[2][EPI_P ? 1 : BM_T][K_STR];
    __shared__ __align__(16) _Float16 smB_[2][EPI_P ? 1 : BN_T][K_STR];
    _Float16 (&smA)[2][BM_T][K_STR] = *reinterpret_cast<_Float16 (*)[2][BM_T][K_STR]>(
        EPI_P ? static_cast<void*>(sm_raw) : static_cast<void*>(smA_));
    _Float16 (&smB)[2][BN_T][K_STR] = *reinterpret_cast<_Float16 (*)[2][BN_T][K_STR]>(
        EPI_P ? static_cast<void*>(sm_raw + SMA_BYTES) : static_cast<void*>(smB_));
    [[maybe_unused]] _Float16* smC = reinterpret_cast<_Float16*>(sm_raw);
    auto swz = [&](int row, int k) __attribute__((always_inline)) -> int {
        if constexpr (XOR_P) return (((k >> 3) ^ ((row / XP) % XC)) << 3) | (k & 7);
        else return k;
    };

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
    {
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
    const int b_k8 = (tid & (B_TPR - 1)) * B_CW;
    int  b_col[B_NCH];
    bool b_valid[B_NCH];
    int  b_n_g[B_NCH];
#pragma unroll
    for (int c = 0; c < B_NCH; c++) {
        b_col[c] = (tid >> B_COL_SHIFT) + c * B_ROWS;
        const int raw = col0 + b_col[c];
        b_valid[c] = BOUNDS_CHECK ? (raw < N) : true;
        b_n_g[c]   = (BOUNDS_CHECK && !b_valid[c]) ? 0 : raw;
    }

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
    [[maybe_unused]] const unsigned char* b_src_u4[B_NCH] = {};
    [[maybe_unused]] const _Float16*      s_src[B_NCH]    = {};
    [[maybe_unused]] const _Float16*      z_src_f16[B_NCH] = {};
#pragma unroll
    for (int c = 0; c < B_NCH; c++) {
        // 0 means the arrival layout, whose stride carries the ONNX group
        // padding described above; a repacked buffer passes its own.
        const int row_bytes =
            b_row_bytes ? b_row_bytes : (num_groups_k * (group_size >> 1));
        b_src_u4[c] = static_cast<const unsigned char*>(B_data) +
                      (static_cast<size_t>(b_n_g[c]) * row_bytes + (b_k8 >> 1));
        s_src[c] = scales + static_cast<size_t>(b_n_g[c]) * num_groups_k;
        if constexpr (USE_ZEROS)
            z_src_f16[c] = static_cast<const _Float16*>(zeros) +
                           static_cast<size_t>(b_n_g[c]) * num_groups_k;
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
        const int source_row = a_valid[c] ? logical_row : 0;
        a_src[c] = A + static_cast<size_t>(source_row) * lda + a_koff[c];
    }

    [[maybe_unused]] const unsigned char* a_sbase = nullptr;
    [[maybe_unused]] const unsigned char* b_sbase = nullptr;
    [[maybe_unused]] uint32_t a_voff = 0, a_cstep = 0, b_voff = 0, b_cstep = 0;
    if constexpr (UADDR) {
        a_sbase = reinterpret_cast<const unsigned char*>(A + static_cast<size_t>(row0) * lda);
        a_voff  = static_cast<uint32_t>((a_row[0] * lda + a_koff[0]) * 2);
        a_cstep = static_cast<uint32_t>((THR_L / A_ROW_SPLIT) * lda * 2);
        const uint32_t b_row =
            static_cast<uint32_t>(b_row_bytes ? b_row_bytes : num_groups_k * (group_size >> 1));
        b_sbase = static_cast<const unsigned char*>(B_data) + static_cast<size_t>(col0) * b_row;
        b_voff  = static_cast<uint32_t>(b_col[0]) * b_row + (b_k8 >> 1);
        b_cstep = static_cast<uint32_t>(B_ROWS) * b_row;
    }
    auto aSrc = [&](int c) __attribute__((always_inline)) -> const void* {
        if constexpr (UADDR) return a_sbase + static_cast<size_t>(a_voff + c * a_cstep);
        else return a_src[c];
    };
    auto bSrc = [&](int c) __attribute__((always_inline)) -> const void* {
        if constexpr (UADDR) return b_sbase + static_cast<size_t>(b_voff + c * b_cstep);
        else return b_src_u4[c];
    };

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
     * so that global loads overlap with Matrix Core WMMA execution. The
     * register set (the last index of each pf_* array) is a compile-time
     * constant at every use, so the arrays stay in registers.
     */

    uint32_t pf_a[PGR][A_NCH][A_CH / 2];
    [[maybe_unused]] uint32_t pf_b4[PGR][B_NCH][B_U32] = {};

    /* Quant-group state, depth 1.
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
    [[maybe_unused]] uint32_t s_pk[B_NCH]    = {};
    [[maybe_unused]] uint32_t bias_pk[B_NCH] = {};
    // Depth 2: raw per-tile group constants, packed at dequant time.
    [[maybe_unused]] _Float16 s_raw[PGR][B_NCH]  = {};
    [[maybe_unused]] _Float16 zp_raw[PGR][B_NCH] = {};
    [[maybe_unused]] const int gs_shift = __builtin_ctz(group_size > 0 ? group_size : 1);

    // fp16 1024.0. OR-ing a 4-bit value into the mantissa of 1024.0 yields
    // exactly 1024+q, which is the integer-to-fp16 conversion without any
    // v_cvt: see the dequant in storeToShared.
    constexpr uint32_t kMagicPk = 0x64006400u;

    auto zeroPoint = [&](int c, int grp) __attribute__((always_inline)) -> _Float16
    {
        if constexpr (USE_ZEROS) return z_src_f16[c][grp];
        else return (_Float16)8;
    };

    auto packConstants = [&](_Float16 s16, _Float16 zp, uint32_t& s_out, uint32_t& b_out)
        __attribute__((always_inline))
    {
        // Subtracting (1024 + zp) from (1024 + q) is exact in fp16 -- both
        // operands and the result are exactly representable -- so the magic-bias
        // conversion loses no accuracy. Folding it into the scale instead
        // (fma(1024+q, s, -(1024+zp)*s)) would not: the fp16 constant
        // -(1024+zp)*s carries a rounding error of half an ULP at magnitude
        // ~1024*s, which is a few percent of the (q-zp)*s result.
        const _Float16 bias = -((_Float16)1024 + zp);
        const uint16_t s_bits = __builtin_bit_cast(uint16_t, s16);
        const uint16_t b_bits = __builtin_bit_cast(uint16_t, bias);
        s_out = (uint32_t(s_bits) << 16) | s_bits;
        b_out = (uint32_t(b_bits) << 16) | b_bits;
    };

    auto loadGroupConstants = [&](int grp) __attribute__((always_inline))
    {
#pragma unroll
        for (int c = 0; c < B_NCH; c++)
            if (b_valid[c]) {
                // Scale first, then zero point: the order the shipped kernel
                // loads them in, which its instruction schedule depends on.
                const _Float16 s16 = s_src[c][grp];
                const _Float16 zp = zeroPoint(c, grp);
                packConstants(s16, zp, s_pk[c], bias_pk[c]);
            }
    };

    if constexpr (PGR == 1) {
        grp_cur   = b_k8 / group_size;
        grp_k_end = (grp_cur + 1) * group_size;
        loadGroupConstants(grp_cur);
    }

    auto issueLoads = [&](int k_abs, auto SETC) __attribute__((always_inline))
    {
        constexpr int S = decltype(SETC)::value;
#pragma unroll
        for(int c = 0; c < A_NCH; c++)
        {
            const bool ok = BOUNDS_CHECK ? a_valid[c] : true;
            if (ok) {
                if constexpr (A_CH == 8)
                    *reinterpret_cast<u32x4*>(pf_a[S][c]) =
                        *reinterpret_cast<const u32x4*>(aSrc(c));
                else
                    *reinterpret_cast<u32x2*>(pf_a[S][c]) =
                        *reinterpret_cast<const u32x2*>(aSrc(c));
            } else {
#pragma unroll
                for (int e = 0; e < A_CH / 2; e++) pf_a[S][c][e] = 0;
            }
            if constexpr (!UADDR) a_src[c] += BK_T;
        }

#pragma unroll
        for (int c = 0; c < B_NCH; c++) {
            if (b_valid[c]) {
                if constexpr (B_U32 == 1)
                    pf_b4[S][c][0] = *reinterpret_cast<const uint32_t*>(bSrc(c));
                else if constexpr (B_U32 == 2)
                    *reinterpret_cast<u32x2*>(pf_b4[S][c]) =
                        *reinterpret_cast<const u32x2*>(bSrc(c));
                else if constexpr (B_U32 == 3) {
                    *reinterpret_cast<u32x2*>(pf_b4[S][c]) =
                        *reinterpret_cast<const u32x2*>(bSrc(c));
                    pf_b4[S][c][2] = reinterpret_cast<const uint32_t*>(bSrc(c))[2];
                } else
                    *reinterpret_cast<u32x4*>(pf_b4[S][c]) =
                        *reinterpret_cast<const u32x4*>(bSrc(c));
            }
            if constexpr (!UADDR) b_src_u4[c] += BK_T >> 1;
        }
        if constexpr (PGR == 1) {
            // This thread's chunks are contiguous in k starting at a
            // multiple of B_CW, and group_size is a power of two >= 16 >=
            // B_CW for every config that sets B_CW > 8, so each shares one
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
        } else {
            const int grp = k_abs >> gs_shift;
#pragma unroll
            for (int c = 0; c < B_NCH; c++)
                if (b_valid[c]) {
                    s_raw[S][c]  = s_src[c][grp];
                    zp_raw[S][c] = zeroPoint(c, grp);
                }
        }
        if constexpr (UADDR) {
            a_sbase += BK_T * 2;
            b_sbase += BK_T >> 1;
        }
    };

    auto storeToShared = [&](int buf, auto SETC) __attribute__((always_inline))
    {
        constexpr int S = decltype(SETC)::value;
#pragma unroll
        for (int c = 0; c < B_NCH; c++) {
            // The destination is formed at each store, as in the shipped
            // kernel; hoisting it changed the dequant/WMMA interleave.
#define MNX_SMB_DST(w) (*reinterpret_cast<u32x4*>(&smB[buf][b_col[c]][swz(b_col[c], b_k8 + 8 * (w))]))
            if (b_valid[c]) {
                /* uint4 -> fp16 without a single v_cvt.
                 *
                 * OR-ing a nibble into the mantissa of fp16 1024.0 gives exactly
                 * 1024+q, and v_perm_b32 places two nibbles into the low bytes
                 * of the two 16-bit lanes in one instruction. Per 8 weights that
                 * is 3 mask ops + 4 perms + 4 ORs + 4 pk_add + 4 pk_mul, against
                 * 8 extracts + 8 v_cvt_f32_ubyte0 + 8 v_cvt_f16_f32 + 4 packs +
                 * 4 pk_fma for the arithmetic conversion the compiler generates
                 * from static_cast<_Float16>(unsigned). */
                uint32_t sp, bp;
                if constexpr (PGR == 1) {
                    sp = s_pk[c];
                    bp = bias_pk[c];
                } else {
                    packConstants(s_raw[S][c], zp_raw[S][c], sp, bp);
                }
                const half2v bias2 = __builtin_bit_cast(half2v, bp);
                const half2v scale2 = __builtin_bit_cast(half2v, sp);
#pragma unroll
                for (int w = 0; w < B_U32; w++) {
                    const uint32_t packed = pf_b4[S][c][w];
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
                    MNX_SMB_DST(w) = out;
                }
            } else {
                const u32x4 zero4 = {0, 0, 0, 0};
#pragma unroll
                for (int w = 0; w < B_U32; w++) MNX_SMB_DST(w) = zero4;
            }
#undef MNX_SMB_DST
        }

#pragma unroll
        for(int c = 0; c < A_NCH; c++)
        {
            _Float16* dst = &smA[buf][a_row[c]][swz(a_row[c], a_koff[c])];
            if constexpr (A_CH == 8)
                *reinterpret_cast<u32x4*>(dst) = *reinterpret_cast<const u32x4*>(pf_a[S][c]);
            else
                *reinterpret_cast<u32x2*>(dst) = *reinterpret_cast<const u32x2*>(pf_a[S][c]);
        }
    };

    // Fragments of one 16-wide K slice: WT_N B fragments and WT_M A fragments,
    // each two 16-byte LDS reads (swizzled when XOR_P).
    auto loadB = [&](int buf, int ks, int wn, half16& f) __attribute__((always_inline))
    {
        const int k_off = hipdnn_wmma_k_off(sub);
        const int row = (wcol * WT_N + wn) * WMMA_TILE + lane;
        u32x4* dst = reinterpret_cast<u32x4*>(&f);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; i++)
            dst[i] = *reinterpret_cast<const u32x4*>(
                &smB[buf][row][swz(row, ks * WMMA_TILE + k_off + 8 * i)]);
    };
    auto loadA = [&](int buf, int ks, int wm, half16& f) __attribute__((always_inline))
    {
        const int k_off = hipdnn_wmma_k_off(sub);
        const int row = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4* dst = reinterpret_cast<u32x4*>(&f);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; i++)
            dst[i] = *reinterpret_cast<const u32x4*>(
                &smA[buf][row][swz(row, ks * WMMA_TILE + k_off + 8 * i)]);
    };

    // ks_begin/ks_end let the caller run the K-step's WMMA in two pieces and put
    // the shared-memory store for the *next* tile between them.
    auto computeWMMA = [&](int buf, int ks_begin, int ks_end)
        __attribute__((always_inline))
    {
#pragma unroll
        for (int ks = ks_begin; ks < ks_end; ks++) {
            half16 b_frag[WT_N];
#pragma unroll
            for (int wn = 0; wn < WT_N; wn++) loadB(buf, ks, wn, b_frag[wn]);
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++) {
                half16 a_frag;
                loadA(buf, ks, wm, a_frag);
#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    /* Interleave pattern for the store-first step: per WMMA, a share of the
     * step's LDS reads, LDS writes, global loads and dequant VALU. Counts are
     * the step's instruction mix estimated from the tile geometry. */
    auto schedPattern = [&]() __attribute__((always_inline))
    {
        constexpr int NW   = K_STEPS * WT_M * WT_N;
        constexpr int NDSR = K_STEPS * (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
        constexpr int NDSW = A_NCH + B_NCH * B_U32;
        constexpr int NVM  = A_NCH + B_NCH * (B_U32 == 3 ? 2 : 1);
        constexpr int NVA  = B_NCH * B_U32 * 19;
        constexpr int R  = (NDSR + NW - 1) / NW;
        constexpr int Wd = (NDSW + NW - 1) / NW;
        constexpr int V  = (NVM + NW - 1) / NW;
        constexpr int VA = (NVA + NW - 1) / NW;
        constexpr bool uR = SCHED_P == 1 || SCHED_P == 2, uW = SCHED_P == 1;
        constexpr bool uV = SCHED_P == 1 || SCHED_P == 2, uVA = SCHED_P == 1 || SCHED_P == 3;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);           // WMMA
            if constexpr (uR && R)   __builtin_amdgcn_sched_group_barrier(0x100, R, 0);   // DS read
            if constexpr (uW && Wd)  __builtin_amdgcn_sched_group_barrier(0x200, Wd, 0);  // DS write
            if constexpr (uV && V)   __builtin_amdgcn_sched_group_barrier(0x020, V, 0);   // VMEM read
            if constexpr (uVA && VA) __builtin_amdgcn_sched_group_barrier(0x002, VA, 0);  // VALU
        }
    };

    using Set0 = std::integral_constant<int, 0>;
    using Set1 = std::integral_constant<int, 1>;

    // issueLoads takes this thread's absolute k, which is what the quant-group
    // tracking keys on; the source pointers advance themselves.
    issueLoads(b_k8, Set0{});
    storeToShared(0, Set0{});

    /* The store stays after the whole WMMA sequence.
     *
     * Splitting the sequence and dequantizing the next tile in the middle looks
     * like it should hide the dequant behind Matrix Core work, and it was tried:
     * it costs 3-38% depending on the tile, because the WT_N fragments have to
     * stay live across the store and the extra pressure spills into worse
     * scheduling. The compiler already interleaves the tail of the WMMA sequence
     * with the dequant on its own. */
    int buf = 0;
    if constexpr (PIPE_SF) {
        /* Store-first. Tile i is computed from LDS buffer i % 2. Each step
         * stores tile i+1 (loaded one step earlier) into the other buffer,
         * issues tile i+2 into the single register set, then runs tile i's
         * WMMAs; the barrier that ends the step makes both buffers safe for the
         * next one. Buffer indices are compile-time constants (two-step unroll)
         * so the compiler can see tile i+1's LDS writes never alias tile i's
         * reads and interleave them. Steps without a tile i+2 are peeled. */
        const int n_tiles = (K + BK_T - 1) / BK_T;
        if (n_tiles > 1) issueLoads(BK_T + b_k8, Set0{});
        HIPDNN_LDS_BARRIER();
        using B0 = std::integral_constant<int, 0>;
        using B1 = std::integral_constant<int, 1>;
        auto sf = [&](int i, auto BUF, auto ISSUE) __attribute__((always_inline))
        {
            constexpr int Bc = decltype(BUF)::value;
            storeToShared(1 - Bc, Set0{});
            if constexpr (decltype(ISSUE)::value) issueLoads((i + 2) * BK_T + b_k8, Set0{});
            computeWMMA(Bc, 0, K_STEPS);
            if constexpr (SCHED_P) schedPattern();
            HIPDNN_LDS_BARRIER();
        };
        int i = 0;
        for (; i + 1 <= n_tiles - 3; i += 2) {
            sf(i, B0{}, std::true_type{});
            sf(i + 1, B1{}, std::true_type{});
        }
        if (i <= n_tiles - 3) {
            sf(i, B0{}, std::true_type{});
            i++;
        }
        if (i == n_tiles - 2) {
            if (i & 1) sf(i, B1{}, std::false_type{});
            else       sf(i, B0{}, std::false_type{});
            i++;
        }
        buf = i & 1;
    } else if constexpr (PGR == 1) {
        HIPDNN_LDS_BARRIER();
        for(int t = BK_T; t < K; t += BK_T)
        {
            int nxt = 1 - buf;
            issueLoads(t + b_k8, Set0{});
            computeWMMA(buf, 0, K_STEPS);
            storeToShared(nxt, Set0{});
            HIPDNN_LDS_BARRIER();
            buf = nxt;
        }
    } else {
        /* Depth 2: each step issues the tile after next into the other
         * register set before running the WMMAs, so two tiles are in flight
         * across each compute phase. */
        if (BK_T < K) issueLoads(BK_T + b_k8, Set1{});
        HIPDNN_LDS_BARRIER();
        /* Tile i (k = i*BK) lives in register set i % 2. Every step but the
         * last issues tile i+1 unconditionally, so neither set ever has to
         * carry a stale value across a conditional; the last step is peeled.
         * The body is unrolled by two to keep the set index constant. */
        auto step = [&](int t, auto SETC, auto ISSUE) __attribute__((always_inline))
        {
            constexpr int S = decltype(SETC)::value;
            const int nxt = 1 - buf;
            if constexpr (decltype(ISSUE)::value)
                issueLoads(t + BK_T + b_k8, std::integral_constant<int, 1 - S>{});
            computeWMMA(buf, 0, K_STEPS);
            storeToShared(nxt, SETC);
            HIPDNN_LDS_BARRIER();
            buf = nxt;
        };
        using Yes = std::true_type;
        using No = std::false_type;
        const int n_tiles = (K + BK_T - 1) / BK_T;
        int i = 1;
        for (; i + 1 <= n_tiles - 2; i += 2) {
            step(i * BK_T, Set1{}, Yes{});
            step((i + 1) * BK_T, Set0{}, Yes{});
        }
        if (i <= n_tiles - 2) {
            step(i * BK_T, Set1{}, Yes{});
            i++;
        }
        if (i == n_tiles - 1) {
            if (i & 1) step(i * BK_T, Set1{}, No{});
            else       step(i * BK_T, Set0{}, No{});
        }
    }
    computeWMMA(buf, 0, K_STEPS);

    if constexpr (EPI_P) {
    /* Staged epilogue: each wave writes its accumulators into an fp16 C tile
     * in LDS (reusing the A/B buffers), then the workgroup copies whole rows to
     * C as 16-byte stores. Interior tiles with a 16-byte-aligned row stride
     * only; edge tiles use the per-lane stores below. */
    if constexpr (!BOUNDS_CHECK) {
        if ((ldc & 7) == 0) {
            constexpr int CV = BN_T / 8;                    // 16-byte vectors per C row
            constexpr int VPT = (BM_T * CV) / THR_L;
            static_assert((BM_T * CV) % THR_L == 0, "C copy must divide the threads");
            HIPDNN_LDS_BARRIER();
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++)
#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const int r = (wrow * WT_M + wm) * WMMA_TILE + hipdnn_wmma_acc_row(sub, e);
                        const int cc = (wcol * WT_N + wn) * WMMA_TILE + lane;
                        smC[r * C_STR + cc] = (_Float16)acc[wm][wn][e];
                    }
            HIPDNN_LDS_BARRIER();
#pragma unroll
            for (int v = 0; v < VPT; v++) {
                const int idx = tid + v * THR_L;
                const int r = idx / CV;
                const int cv = idx % CV;
                const u32x4 val = *reinterpret_cast<const u32x4*>(&smC[r * C_STR + cv * 8]);
                *reinterpret_cast<u32x4*>(&C[static_cast<size_t>(row0 + r) * ldc + col0 + cv * 8]) = val;
            }
            return;
        }
    }
    }

#pragma unroll
    for(int wm = 0; wm < WT_M; wm++)
    {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for(int wn = 0; wn < WT_N; wn++)
        {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
            // One row pointer per 16x16 tile, then a compile-time row offset
            // times ldc per element, as in GemmFp16U4Impl: written as
            // C[(mbase + r) * ldc + ...], the compiler recomputes the whole
            // 32-bit index and does a 64-bit sign-extend, shift and add for
            // every stored element.
            const int r0 = hipdnn_wmma_acc_row(sub, 0);
            _Float16* crow = C + static_cast<int64_t>(mbase + r0) * ldc + nbase + lane;
#pragma unroll
            for(int e = 0; e < 8; e++)
            {
                int r = hipdnn_wmma_acc_row(sub, e);
                _Float16* cp = crow + static_cast<int64_t>(r - r0) * ldc;
                if (!BOUNDS_CHECK || (mbase + r < M && nbase + lane < N))
                    *cp = (_Float16)acc[wm][wn][e];
            }
        }
    }
}


#endif // HIPDNN_MATMUL_NBITS_WMMA_CORE_X_H
