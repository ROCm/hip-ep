/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CUSTOM_KERNELS_WMMA_FRAG_H
#define HIP_CUSTOM_KERNELS_WMMA_FRAG_H

/*
 * Per-arch WMMA helpers for the 16x16x16 FP16 input / FP32 accumulator
 * `v_wmma_f32_16x16x16_f16` instruction.  Two layouts are supported:
 *
 *   - RDNA3 wave32 (gfx11xx).  A,B fragments are <16 x fp16> per lane.
 *     The lower half-wave (lanes 0-15) is replicated into the upper
 *     half-wave (lanes 16-31), so every lane holds the full 16-element
 *     K block for its row (A) or column (B).  C/D fragments are
 *     <8 x f32> per lane; element e maps to matrix row `e*2 + sub`
 *     where `sub = lid / 16`.
 *     Builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
 *
 *   - RDNA4 wave32 (gfx1200/gfx1201).  A,B fragments are <8 x fp16>
 *     per lane with NO half-wave replication.  Each lane (i, sub) holds
 *     the K elements
 *           {4*sub+0, 4*sub+1, 4*sub+2, 4*sub+3,
 *            8+4*sub+0, 8+4*sub+1, 8+4*sub+2, 8+4*sub+3}
 *     of its row (A) or column (B).  C/D fragments stay <8 x f32> per
 *     lane; element e maps to matrix row `e + 8*sub`.
 *     Builtin: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12.
 *
 * Mappings derived from the AMD Matrix Instruction Calculator
 * (https://github.com/ROCm/amd_matrix_instruction_calculator) for
 * `v_wmma_f32_16x16x16_f16` on rdna3 / rdna4 with no modifiers.
 *
 * Helpers:
 *   - `wmma_ab_frag_t`            : per-lane fragment type for A or B.
 *   - `wmma_cd_frag_t`            : per-lane fragment type for C or D.
 *   - `wmma_ab_load_row_major`    : load 8/16 fp16 from a contiguous
 *                                   row buffer (length 16) into an A
 *                                   fragment, indexed by the matrix
 *                                   row `i` and the K-step base.
 *   - `wmma_ab_load_col_major`    : load 8/16 fp16 from a strided
 *                                   column buffer into a B fragment.
 *   - `wmma_cd_row_for_element`   : matrix row index for accumulator
 *                                   element `e` given `sub`.
 *   - `wmma_f32_16x16x16_f16`     : the FP16 -> FP32 WMMA call.
 *
 * The helpers are intended for use from device code only.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 hip_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 hip_half8  __attribute__((ext_vector_type(8)));
typedef float    hip_float8 __attribute__((ext_vector_type(8)));

#if defined(__gfx1200__) || defined(__gfx1201__)
#define HIP_WMMA_GFX12 1
#else
#define HIP_WMMA_GFX12 0
#endif

#if HIP_WMMA_GFX12
typedef hip_half8  wmma_ab_frag_t;
#else
typedef hip_half16 wmma_ab_frag_t;
#endif
typedef hip_float8 wmma_cd_frag_t;

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

/*
 * Decode the wave-relative thread id into the per-lane (i, sub) WMMA
 * coordinates for the A/B/C/D fragments.  `i` is the row of A (or column
 * of B / C) that this lane participates in; `sub` selects the K (or M)
 * sub-slab.
 */
__device__ __forceinline__
void hip_wmma_lane_decode(int lid, int* i, int* sub) {
    *i   = lid % 16;
    *sub = lid / 16;
}

/*
 * Issue the FP16 -> FP32 16x16x16 WMMA on whichever architecture is
 * being compiled for.  Identical signature on both archs.
 */
__device__ __forceinline__
hip_float8 hip_wmma_f32_16x16x16_f16(wmma_ab_frag_t a,
                                     wmma_ab_frag_t b,
                                     hip_float8 c) {
#if HIP_WMMA_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

/*
 * Load an A fragment from a contiguous K-row of length 16.  `row_ptr`
 * must point at the K=0 element of the matrix row that this lane
 * participates in.  On RDNA3 every lane copies all 16 halves; on
 * RDNA4 each lane copies only the 8 halves that the gfx12 layout
 * assigns to it.  Identical layout requirement on the producer side
 * for both archs (a flat row of 16 contiguous fp16).
 */
__device__ __forceinline__
wmma_ab_frag_t hip_wmma_load_ab_row(const _Float16* row_ptr, int sub) {
    wmma_ab_frag_t frag;
#if HIP_WMMA_GFX12
    uint32_t* dst = reinterpret_cast<uint32_t*>(&frag);
    const uint32_t* src_lo = reinterpret_cast<const uint32_t*>(row_ptr + 4 * sub);
    const uint32_t* src_hi = reinterpret_cast<const uint32_t*>(row_ptr + 8 + 4 * sub);
    dst[0] = src_lo[0];
    dst[1] = src_lo[1];
    dst[2] = src_hi[0];
    dst[3] = src_hi[1];
#else
    (void)sub;
    uint32_t* dst = reinterpret_cast<uint32_t*>(&frag);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(row_ptr);
#pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
#endif
    return frag;
}

/*
 * Element-wise A/B fragment fill.  Used by kernels that compute or
 * gather A/B element-by-element instead of doing a row copy.  The
 * caller must supply the K index that this slot of the fragment
 * corresponds to.  Returns the K index (0..15) that fragment slot
 * `e` of lane (sub) corresponds to, so the caller can fill it.
 *
 *   On RDNA3 (no replication needed at the user level since the same
 *   data is written to lane and lane+16 by repeated execution):
 *       slot e of any lane is K = e          (e in [0,16))
 *
 *   On RDNA4 (per-lane unique K elements):
 *       slot e of lane (sub) is K = (e < 4) ? (4*sub + e)
 *                                           : (8 + 4*sub + e - 4)
 */
__device__ __forceinline__
int hip_wmma_ab_k_for_element(int e, int sub) {
#if HIP_WMMA_GFX12
    if (e < 4) return 4 * sub + e;
    return 8 + 4 * sub + (e - 4);
#else
    (void)sub;
    return e;
#endif
}

/*
 * Number of fp16 slots per lane per fragment for the elementwise
 * fill API above.  16 on RDNA3 (with replication), 8 on RDNA4.
 */
__device__ __forceinline__
constexpr int hip_wmma_ab_slots() {
#if HIP_WMMA_GFX12
    return 8;
#else
    return 16;
#endif
}

/*
 * For an accumulator element `e` (0..7) at half-wave id `sub`, return
 * the matrix row index of the C/D entry it maps to.  Column index is
 * always `i = lid % 16`.
 */
__device__ __forceinline__
int hip_wmma_cd_row(int e, int sub) {
#if HIP_WMMA_GFX12
    return e + 8 * sub;
#else
    return e * 2 + sub;
#endif
}

#endif // HIP_CUSTOM_KERNELS_WMMA_FRAG_H
