/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Validation, row expansion and weight-layout rearrangement for the constants
// listed in __metadata_blob. Host-only and HIP-free, so test/runtime exercises
// it natively.
#ifndef HIPDNN_EP_CONSTANT_LAYOUT_H
#define HIPDNN_EP_CONSTANT_LAYOUT_H

#include "hip/matmul_nbits_weight_layout.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace hipdnn_ep {

// Where one constant goes in the constants blob and how much to read for it.
// A row-padded constant's source holds `rows` dense rows of `row_bytes`, which
// the blob holds `row_pitch` apart (size = rows * row_pitch); any other
// constant's source holds `size` bytes. A constant with a weight_layout other
// than Rows holds `rows` dense rows of `row_bytes` (row_pitch == row_bytes),
// which the upload rearranges into that byte order.
struct ConstantLayout {
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t src_size = 0;
  uint64_t row_bytes = 0;
  uint64_t row_pitch = 0;
  uint64_t rows = 0;
  hip::MatmulNBitsWeightLayout weight_layout =
      hip::MatmulNBitsWeightLayout::Rows;
};

// Checks one ConstantInfo and fills `out`. Returns nullptr when it is valid,
// else a description of the rule it breaks.
inline const char *planConstantLayout(int64_t offset, int64_t size,
                                      int64_t row_bytes, int64_t row_pitch,
                                      int64_t weight_layout,
                                      ConstantLayout *out) {
  if (offset < 0 || size < 0)
    return "negative offset or size";
  if (offset > std::numeric_limits<int64_t>::max() - size)
    return "offset + size overflows";
  if (row_bytes < 0 || row_pitch < 0)
    return "negative row_bytes or row_pitch";
  if ((row_bytes == 0) != (row_pitch == 0))
    return "row_bytes and row_pitch must be both zero or both positive";
  ConstantLayout l;
  l.offset = static_cast<uint64_t>(offset);
  l.size = static_cast<uint64_t>(size);
  l.src_size = l.size;
  if (row_pitch != 0) {
    if (row_pitch < row_bytes)
      return "row_pitch is smaller than row_bytes";
    if (size % row_pitch != 0)
      return "size is not a multiple of row_pitch";
    l.row_bytes = static_cast<uint64_t>(row_bytes);
    l.row_pitch = static_cast<uint64_t>(row_pitch);
    l.rows = l.size / l.row_pitch;
    l.src_size = l.rows * l.row_bytes; // <= size, cannot overflow
  }
  if (!hip::isValidMatmulNBitsWeightLayout(weight_layout))
    return "unknown weight_layout";
  l.weight_layout = static_cast<hip::MatmulNBitsWeightLayout>(weight_layout);
  if (l.weight_layout != hip::MatmulNBitsWeightLayout::Rows) {
    if (row_pitch == 0 || row_pitch != row_bytes)
      return "a weight_layout needs row_bytes == row_pitch (no row pad)";
    if (!hip::matmulNBitsWeightLayoutFits(
            l.weight_layout, static_cast<int64_t>(l.rows), row_bytes))
      return "weight_layout is not defined for this row count and row size";
  }
  *out = l;
  return nullptr;
}

// Spreads the `rows` dense rows at the start of `buf` to `row_pitch` apart and
// zero-fills each row's pad. `buf` holds `size` bytes. Last row first, so no
// row is overwritten before it moves.
inline void expandRowsInPlace(uint8_t *buf, const ConstantLayout &l) {
  for (uint64_t r = l.rows; r-- > 0;) {
    std::memmove(buf + r * l.row_pitch, buf + r * l.row_bytes, l.row_bytes);
    std::memset(buf + r * l.row_pitch + l.row_bytes, 0,
                l.row_pitch - l.row_bytes);
  }
}

// Bytes of scratch rearrangeInPlace needs for `l` (0: nothing to rearrange).
inline uint64_t weightLayoutScratchBytes(const ConstantLayout &l) {
  if (l.weight_layout == hip::MatmulNBitsWeightLayout::Rows)
    return 0;
  return hip::matmulNBitsWeightLayoutRowStep(l.weight_layout) * l.row_bytes;
}

// Rearranges the `rows` dense rows at the start of `buf` into the constant's
// weight layout, in place. `scratch` holds weightLayoutScratchBytes(l) bytes.
inline void rearrangeInPlace(uint8_t *buf, const ConstantLayout &l,
                             uint8_t *scratch) {
  hip::applyMatmulNBitsWeightLayout(buf, 0, l.rows, l.row_bytes,
                                    l.weight_layout, scratch);
}

} // namespace hipdnn_ep

#endif // HIPDNN_EP_CONSTANT_LAYOUT_H
