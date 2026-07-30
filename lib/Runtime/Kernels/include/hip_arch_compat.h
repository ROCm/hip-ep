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
 *   HIPDNN_HAS_WMMA   1 only in a device pass for an arch that has the WMMA
 *                     intrinsics (RDNA3/RDNA4). 0 on CDNA and in the host pass,
 *                     so WMMA code paths compile away (and are never launched --
 *                     the host dispatch checks the device warpSize at runtime).
 *
 * ROCm 7.x no longer defines __AMDGCN_WAVEFRONT_SIZE[_], so we key off the
 * clang-provided GPU-family macros (__GFX8__, __GFX9__, __GFX11__, __GFX12__).
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

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
#  define HIPDNN_HAS_WMMA 1
#else
#  define HIPDNN_HAS_WMMA 0
#endif

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

/* Host-side runtime probe: does the current device run wave64 (CDNA)?  Used by
 * the host dispatch to avoid launching WMMA kernels on CDNA, where they compile
 * to a trap. Cached after the first query. */
#ifdef __cplusplus
#include <hip/hip_runtime.h>
static inline bool hipdnn_device_is_wave64() {
  static int cached = -1;
  if (cached < 0) {
    int dev = 0;
    hipGetDevice(&dev);
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, dev) == hipSuccess)
      cached = (p.warpSize >= 64) ? 1 : 0;
    else
      cached = 0;  // assume wave32/WMMA-capable on query failure (RDNA default)
  }
  return cached == 1;
}
#endif

#endif  /* HIP_ARCH_COMPAT_H */
