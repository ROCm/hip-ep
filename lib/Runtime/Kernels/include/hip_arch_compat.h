/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Wavefront-size / matrix-intrinsic portability shim.
 *
 * The custom kernels were originally written for RDNA3/RDNA4 (gfx11xx/gfx12xx),
 * which run wave32 and expose the WMMA matrix intrinsics
 * (__builtin_amdgcn_wmma_*). CDNA (gfx9xx, e.g. MI300/MI350) runs wave64 and has
 * no WMMA -- it uses MFMA instead. This header lets the shared kernel sources
 * compile for both families:
 *
 *   HIPDNN_WAVE_SIZE  compile-time wavefront size in the *device* pass
 *                     (64 on CDNA/gfx9xx, 32 on RDNA/gfx10xx+). Used to make
 *                     warp-shuffle reductions span the whole wave correctly.
 *   HIPDNN_HAS_WMMA   1 only in a device pass for an arch that has one of the
 *                     WMMA f16 16x16x16 builtins below (RDNA3/RDNA3.5/RDNA4).
 *                     0 on CDNA and in the host pass, so WMMA code paths
 *                     compile away (and are never launched -- the host
 *                     dispatch checks the device warpSize at runtime).
 *   HIPDNN_WMMA_GFX12 1 when only the newer, gfx12-style WMMA encoding is
 *                     available (see below). Selects the fragment layout the
 *                     kernels must use.
 *
 * ROCm 7.x no longer defines __AMDGCN_WAVEFRONT_SIZE[_], so we key off the
 * clang-provided GPU-family macros (__GFX8__, __GFX9__, __GFX11__, __GFX12__)
 * for the wave size. WMMA support/layout, however, is NOT reliably keyed off
 * those family macros: gfx1170 ("RDNA 4m") is numbered under the gfx11 family
 * (__GFX11__ is defined, __GFX12__ is not) but its hardware does not implement
 * the original gfx11 WMMA encoding (WMMA256bInsts) at all -- only the newer,
 * unified encoding introduced for gfx12 (WMMA128bInsts). Concretely, on this
 * target __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32) is false
 * and __has_builtin(..._w32_gfx12) is true, even though __GFX11__ is set.
 * Verified by compiling a probe for each target with AMD clang 23.0.0git:
 *   gfx1100/1102/1150/1151/gfx11-generic -> old builtin only, __GFX11__
 *   gfx1170                              -> new (_gfx12) builtin only, __GFX11__
 *   gfx1200/1201/gfx12-generic           -> new (_gfx12) builtin only, __GFX12__
 * So WMMA capability/layout must be probed via __has_builtin against the
 * *builtin names*, not the family macros -- that automatically resolves per
 * the actual -offload-arch of each object in a multi-arch build, which is
 * exactly the "pick the right code per machine" behavior HIP's fat-binary
 * loader then dispatches at load time.
 *
 * The two encodings differ in exactly three ways (see gemm_wmma_kernel.hip /
 * gqa_kernel.hip for the empirically-verified per-lane formulas):
 *   - fragment width: 16 elements/lane (gfx11, K replicated across lane/16)
 *     vs 8 elements/lane (gfx12-style, K split by lane/16, no replication)
 *   - the builtin itself (plain vs "_gfx12" suffix)
 *   - accumulator row mapping: e*2+pair (gfx11, interleaved) vs pair*8+e
 *     (gfx12-style, blocked). The M/N<->lane mapping (row=lane%16 for A,
 *     col=lane%16 for B/output) is identical between the two encodings.
 */
#ifndef HIP_ARCH_COMPAT_H
#define HIP_ARCH_COMPAT_H

#if defined(__HIP_DEVICE_COMPILE__)
#  if defined(__GFX8__) || defined(__GFX9__)
#    define HIPDNN_WAVE_SIZE 64
#  else
#    define HIPDNN_WAVE_SIZE 32
#  endif
#else
/* Host pass: value is unused inside __global__ bodies, but must be a valid
 * compile-time constant for shared-memory sizing etc. */
#  define HIPDNN_WAVE_SIZE 32
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                        \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
#  define HIPDNN_HAS_WMMA 1
#  define HIPDNN_WMMA_GFX12 0
#elif defined(__HIP_DEVICE_COMPILE__) &&                                      \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#  define HIPDNN_HAS_WMMA 1
#  define HIPDNN_WMMA_GFX12 1
#else
#  define HIPDNN_HAS_WMMA 0
#  define HIPDNN_WMMA_GFX12 0
#endif

/* Fragment element count for one A/B operand of the f16 16x16x16 WMMA op:
 * the whole K=16 (replicated across the wave's two lane-halves) on gfx11, or
 * half of K=16 (no replication, split by lane/16) on the gfx12-style
 * encoding. The accumulator is always 8 elements/lane in both encodings. */
#define HIPDNN_WMMA_FRAG_ELEMS (HIPDNN_WMMA_GFX12 ? 8 : 16)

#if HIPDNN_WMMA_GFX12
#  define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                               \
     __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))
#else
#  define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                               \
     __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))
#endif

#if defined(__HIPCC__)
/* __device__/__forceinline__ come from hip_runtime.h; some callers (e.g.
 * matmul_nbits_kernel.hip) include this header before it, so pull it in here
 * too rather than depending on include order. hipRTC is the exception: it
 * injects its own preamble and the path does not resolve. */
#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif
/* K-axis base offset (within a 16-wide K tile) that a fragment load must add
 * for this lane's pair-half (pair = lane/16). Zero on gfx11 (each lane loads
 * the whole 16-wide K range); pair*8 on the gfx12-style encoding (each lane
 * loads only its half). */
__device__ __forceinline__ int hipdnn_wmma_k_off(int pair) {
#if HIPDNN_WMMA_GFX12
  return pair * 8;
#else
  (void)pair;
  return 0;
#endif
}

/* Row contributed by accumulator element `e` (0..7) for this lane's pair-half
 * (pair = lane/16): interleaved (e*2+pair) on gfx11, blocked (pair*8+e) on
 * the gfx12-style encoding. col is lane%16 in both encodings (unchanged). */
__device__ __forceinline__ int hipdnn_wmma_acc_row(int pair, int e) {
#if HIPDNN_WMMA_GFX12
  return pair * 8 + e;
#else
  return e * 2 + pair;
#endif
}
#endif  /* __HIPCC__ */

/* Portable 4x8-bit integer dot-product with i32 accumulate (a signed, b treated
 * as small non-negative bytes -- e.g. unpacked 4-bit weight nibbles 0..15, or
 * the 0x01010101 mask used to sum a quantized activation vector).
 *
 * RDNA3/RDNA4 (gfx11xx/gfx12xx) expose the mixed-sign v_dot4_i32_iu8 via
 * __builtin_amdgcn_sudot4 (dot8-insts). CDNA (gfx9xx, e.g. MI300/MI350) has no
 * dot8-insts but does provide the signed v_dot4_i32_i8
 * (__builtin_amdgcn_sdot4). Because every `b` operand here is a byte in [0,127],
 * the signed and unsigned interpretations are numerically identical, so the CDNA
 * signed intrinsic is a drop-in. The scalar branch keeps the host pass (and any
 * arch without dot instructions) compilable; it is never used for codegen on the
 * supported GPUs. Only defined in a HIP translation unit. */
#if defined(__HIPCC__)
__device__ static inline int hipdnn_sudot4(int a, int b, int acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
  return __builtin_amdgcn_sudot4(true, a, false, b, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
  int r = acc;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int av = static_cast<int>(static_cast<signed char>((a >> (i * 8)) & 0xFF));
    const int bv = static_cast<int>(static_cast<unsigned char>((b >> (i * 8)) & 0xFF));
    r += av * bv;
  }
  return r;
#endif
}
#endif  /* __HIPCC__ */

/* Host-side runtime probe for the wavefront size of the *currently selected*
 * device. Used by the host dispatch to avoid launching WMMA kernels on CDNA
 * (where they compile to a trap) and to size blocks as whole waves.
 *
 * The result is cached per device ordinal rather than once per process: a host
 * may drive several GPUs of different families (e.g. a gfx1151 APU alongside a
 * discrete card), and a single cached answer would then be applied to the wrong
 * device after hipSetDevice. Query failure yields 32, the RDNA/WMMA-capable
 * default, so dispatch decisions stay as they were before CDNA support.
 *
 * Excluded from hipRTC: it compiles device code only, and the HIP host API this
 * needs is not available there. */
#if defined(__cplusplus) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
static inline int hipdnn_device_wave_size() {
  constexpr int kMaxCachedDevices = 16;
  // Zero-initialized (0 == not probed yet), so no thread-safe-statics guard is
  // emitted; concurrent probes race only to write the same value.
  static int cached[kMaxCachedDevices] = {};

  auto probe = [](int d) {
    hipDeviceProp_t p;
    return (hipGetDeviceProperties(&p, d) == hipSuccess && p.warpSize >= 64)
               ? 64
               : 32;
  };

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0)
    return 32;
  if (dev >= kMaxCachedDevices)
    return probe(dev);

  int w = cached[dev];
  if (w == 0) {
    w = probe(dev);
    cached[dev] = w;
  }
  return w;
}

static inline bool hipdnn_device_is_wave64() {
  return hipdnn_device_wave_size() >= 64;
}
#endif

#endif  /* HIP_ARCH_COMPAT_H */
