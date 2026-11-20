/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_MATMUL_NBITS_ROW_ALIAS_H
#define HIP_MATMUL_NBITS_ROW_ALIAS_H

#include <cstdint>

namespace hip {

/* True when the arrival row stride K/2 of a 4-bit MatMulNBits weight falls in a
 * cache-aliasing regime.
 *
 * Rows a tile reads in one K-step sit stride_lines = (K/2)/128 cache lines
 * apart, which reaches only 64 / gcd(stride_lines, 64) of the 64 sets. When
 * that GCD is large -- many factors of two in K/2 -- the rows collide and evict
 * each other even though the working set fits in cache. Requiring the stride to
 * be a whole number of lines makes stride_lines exact, and a GCD of at least 8
 * means at most 8 reachable sets, which is where the measured 30-89% slowdowns
 * start.
 *
 * Fires for K=4096 (stride 2048, 16 lines, gcd 16) and K=8192 (gcd 32). Does
 * not for K=2880 (stride 1440, not a whole number of lines) or K=3584 (14
 * lines, gcd 2). The two conditions together imply K % 2048 == 0.
 *
 * The compiler pads such weights at load time and the runtime repacks them per
 * call when the compiler did not; both must make the same decision, so this is
 * the only definition. */
constexpr bool matmulNBitsRowStrideAliases(int64_t k) {
  const int64_t stride = k / 2;
  if (stride % 128 != 0)
    return false;
  int64_t a = stride / 128, b = 64;
  while (b != 0) {
    const int64_t t = b;
    b = a % b;
    a = t;
  }
  return a >= 8;
}

} // namespace hip

#endif // HIP_MATMUL_NBITS_ROW_ALIAS_H
