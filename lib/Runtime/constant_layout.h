/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Validation and row expansion for the constants listed in __metadata_blob.
// Host-only and HIP-free, so test/runtime exercises it natively.
#ifndef HIPDNN_EP_CONSTANT_LAYOUT_H
#define HIPDNN_EP_CONSTANT_LAYOUT_H

#include <cstdint>
#include <cstring>
#include <limits>

namespace hipdnn_ep {

// Where one constant goes in the constants blob and how much to read for it.
// A row-padded constant's source holds `rows` dense rows of `row_bytes`, which
// the blob holds `row_pitch` apart (size = rows * row_pitch); any other
// constant's source holds `size` bytes.
struct ConstantLayout {
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t src_size = 0;
  uint64_t row_bytes = 0;
  uint64_t row_pitch = 0;
  uint64_t rows = 0;
};

// Checks one ConstantInfo and fills `out`. Returns nullptr when it is valid,
// else a description of the rule it breaks.
inline const char *planConstantLayout(int64_t offset, int64_t size,
                                      int64_t row_bytes, int64_t row_pitch,
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

} // namespace hipdnn_ep

#endif // HIPDNN_EP_CONSTANT_LAYOUT_H
