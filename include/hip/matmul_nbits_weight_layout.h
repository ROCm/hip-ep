/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Byte orders an int4 MatMulNBits weight can be stored in, shared by the
// compiler (hip-pad-matmul-nbits-weights, hip-externalize-constants), the
// constants writer, the runtime loader and the kernels. Header-only and free of
// MLIR, LLVM and HIP so every one of them includes it.
//
// Every order keeps the dense size, N rows of `row_bytes` = K/2 bytes, and only
// moves bytes within a row (XOR lines) or within a group of 16 rows (blocks),
// so a weight can be rearranged in place, one row or row group at a time:
//
//   Rows      row n at n * row_bytes (the arrival layout).
//   XorLines  the 128-byte lines of row n stored at line l ^ f(n), where
//             f(n) = (n / R) % P, P is the largest power of two dividing the
//             line count L = row_bytes / 128 and R = 64 / gcd(L, 64). Rows R
//             apart then start in different cache sets. Identity when
//             row_bytes is not a whole number of lines.
//   Blocks    512-byte blocks of 16 rows x 32 bytes (64 int4 values), block
//             (n / 16, j) at (n / 16 * (row_bytes / 32) + j) * 512, row n % 16
//             of it at 32 * (n % 16). One 16-byte load per lane of a wave
//             reads a whole block. Needs N % 16 == 0 and row_bytes % 32 == 0.
#ifndef HIP_MATMUL_NBITS_WEIGHT_LAYOUT_H
#define HIP_MATMUL_NBITS_WEIGHT_LAYOUT_H

#include <cstdint>
#include <cstring>

namespace hip {

enum class MatmulNBitsWeightLayout : int32_t {
  Rows = 0,
  XorLines = 1,
  Blocks = 2,
};

inline bool isValidMatmulNBitsWeightLayout(int64_t layout) {
  return layout >= 0 && layout <= 2;
}

/// Whether `layout` is defined for N rows of `rowBytes` bytes.
inline bool matmulNBitsWeightLayoutFits(MatmulNBitsWeightLayout layout,
                                        int64_t rows, int64_t rowBytes) {
  if (rows <= 0 || rowBytes <= 0)
    return false;
  switch (layout) {
  case MatmulNBitsWeightLayout::Rows:
  case MatmulNBitsWeightLayout::XorLines:
    return true;
  case MatmulNBitsWeightLayout::Blocks:
    return rows % 16 == 0 && rowBytes % 32 == 0;
  }
  return false;
}

/// XorLines: the XOR applied to row n's line index (0 for every row when the
/// row is not a whole number of 128-byte lines).
inline uint32_t matmulNBitsXorLineMask(uint64_t n, uint64_t rowBytes) {
  if (rowBytes % 128 != 0)
    return 0;
  const uint64_t lines = rowBytes / 128;
  const uint64_t p = lines & (~lines + 1);
  const uint64_t capped = p < 64 ? p : 64;
  unsigned shift = 6;
  for (uint64_t v = capped; v > 1; v >>= 1)
    --shift;
  return static_cast<uint32_t>((n >> shift) & (p - 1));
}

/// Number of consecutive rows each in-place step moves together.
inline uint64_t matmulNBitsWeightLayoutRowStep(MatmulNBitsWeightLayout layout) {
  return layout == MatmulNBitsWeightLayout::Blocks ? 16 : 1;
}

/// Rearranges rows [firstRow, firstRow + rows) of a dense weight, held at
/// `buf`, into `layout`. `rows` and `firstRow` are multiples of the layout's
/// row step; `scratch` holds rowStep * rowBytes bytes. Rows is a no-op.
inline void applyMatmulNBitsWeightLayout(uint8_t *buf, uint64_t firstRow,
                                         uint64_t rows, uint64_t rowBytes,
                                         MatmulNBitsWeightLayout layout,
                                         uint8_t *scratch) {
  switch (layout) {
  case MatmulNBitsWeightLayout::Rows:
    return;
  case MatmulNBitsWeightLayout::XorLines: {
    if (rowBytes % 128 != 0)
      return;
    const uint64_t lines = rowBytes / 128;
    for (uint64_t r = 0; r < rows; ++r) {
      const uint32_t f = matmulNBitsXorLineMask(firstRow + r, rowBytes);
      if (f == 0)
        continue;
      uint8_t *row = buf + r * rowBytes;
      std::memcpy(scratch, row, rowBytes);
      for (uint64_t l = 0; l < lines; ++l)
        std::memcpy(row + ((l ^ f) << 7), scratch + (l << 7), 128);
    }
    return;
  }
  case MatmulNBitsWeightLayout::Blocks: {
    const uint64_t pieces = rowBytes / 32;
    const uint64_t groupBytes = 16 * rowBytes;
    for (uint64_t g = 0; g < rows / 16; ++g) {
      uint8_t *grp = buf + g * groupBytes;
      std::memcpy(scratch, grp, groupBytes);
      for (uint64_t n = 0; n < 16; ++n)
        for (uint64_t j = 0; j < pieces; ++j)
          std::memcpy(grp + j * 512 + n * 32, scratch + n * rowBytes + j * 32,
                      32);
    }
    return;
  }
  }
}

} // namespace hip

#endif // HIP_MATMUL_NBITS_WEIGHT_LAYOUT_H
