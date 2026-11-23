/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_KQUANT_DOT_H
#define HIP_KQUANT_DOT_H

// The K-quant kernels multiply signed activation bytes by non-negative
// packed-weight bytes. RDNA exposes the mixed-sign dot instruction directly;
// CDNA can use the signed instruction because every weight byte is in [0, 127].
#if defined(__HIPCC__)
__device__ static inline int kquant_sudot4(int a, int b, int acc) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__GFX11__) || defined(__GFX12__))
  return __builtin_amdgcn_sudot4(true, a, false, b, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
  return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
  int result = acc;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int lhs =
        static_cast<int>(static_cast<signed char>((a >> (index * 8)) & 0xff));
    const int rhs =
        static_cast<int>(static_cast<unsigned char>((b >> (index * 8)) & 0xff));
    result += lhs * rhs;
  }
  return result;
#endif
}
#endif

#endif // HIP_KQUANT_DOT_H
