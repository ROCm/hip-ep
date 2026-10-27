/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// planConstantLayout / expandRowsInPlace (lib/Runtime/constant_layout.h): the
// checks the runtime applies to every ConstantInfo before it allocates or
// copies, and the in-place expansion of dense source rows to the stored row
// pitch.

#include "constant_layout.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using hipdnn_ep::ConstantLayout;
using hipdnn_ep::expandRowsInPlace;
using hipdnn_ep::planConstantLayout;

static int g_failures = 0;

#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static void testDense() {
  ConstantLayout l;
  EXPECT(planConstantLayout(64, 1000, 0, 0, &l) == nullptr);
  EXPECT(l.offset == 64 && l.size == 1000 && l.src_size == 1000);
  EXPECT(l.rows == 0 && l.row_pitch == 0);
  EXPECT(planConstantLayout(0, 0, 0, 0, &l) == nullptr);
}

// Expands `rows` rows of `row_bytes` (row r filled with value r + 1) to
// `row_pitch`, with guard bytes on both sides of the buffer.
static void checkExpansion(int64_t rows, int64_t row_bytes, int64_t row_pitch) {
  ConstantLayout l;
  EXPECT(planConstantLayout(0, rows * row_pitch, row_bytes, row_pitch, &l) ==
         nullptr);
  EXPECT(l.rows == static_cast<uint64_t>(rows));
  EXPECT(l.src_size == static_cast<uint64_t>(rows * row_bytes));
  constexpr size_t kGuard = 32;
  constexpr uint8_t kGuardByte = 0xA5;
  std::vector<uint8_t> mem(kGuard + l.size + kGuard, kGuardByte);
  uint8_t *buf = mem.data() + kGuard;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t b = 0; b < row_bytes; ++b)
      buf[r * row_bytes + b] = static_cast<uint8_t>(r + 1);
  // The staging buffer holds `size` bytes; bytes past the dense rows are
  // whatever the previous constant left there.
  for (uint64_t i = l.src_size; i < l.size; ++i)
    buf[i] = 0xEE;
  expandRowsInPlace(buf, l);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < row_bytes; ++b)
      EXPECT(buf[r * row_pitch + b] == static_cast<uint8_t>(r + 1));
    for (int64_t b = row_bytes; b < row_pitch; ++b)
      EXPECT(buf[r * row_pitch + b] == 0);
  }
  for (size_t i = 0; i < kGuard; ++i) {
    EXPECT(mem[i] == kGuardByte);
    EXPECT(mem[kGuard + l.size + i] == kGuardByte);
  }
}

static void testRowPadded() {
  checkExpansion(2, 48, 64);
  checkExpansion(37, 2048, 2304);
  checkExpansion(1, 16, 272);
  // row_pitch == row_bytes is a dense-equivalent layout: nothing moves.
  checkExpansion(5, 128, 128);
}

static void testMalformed() {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  ConstantLayout l;
  EXPECT(planConstantLayout(-1, 64, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, -64, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(kMax - 8, 64, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, -16, 64, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, 16, -64, &l) != nullptr);
  // One row field set without the other.
  EXPECT(planConstantLayout(0, 128, 0, 64, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, 64, 0, &l) != nullptr);
  // Pitch smaller than the row.
  EXPECT(planConstantLayout(0, 128, 64, 32, &l) != nullptr);
  // Footprint not a whole number of rows (would leave the tail unwritten),
  // including a pitch larger than the footprint.
  EXPECT(planConstantLayout(0, 130, 48, 64, &l) != nullptr);
  EXPECT(planConstantLayout(0, 32, 16, 64, &l) != nullptr);
}

int main() {
  testDense();
  testRowPadded();
  testMalformed();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("constant layout: all checks passed\n");
  return 0;
}
