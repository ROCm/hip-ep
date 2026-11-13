/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// planConstantLayout / expandRowsInPlace / rearrangeInPlace
// (lib/Runtime/constant_layout.h): the checks the runtime applies to every
// ConstantInfo before it allocates or copies, the in-place expansion of dense
// source rows to the stored row pitch, and the in-place rearrangement into a
// weight layout, checked byte by byte against where each layout puts a byte.

#include "constant_layout.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using hipdnn_ep::ConstantLayout;
using hipdnn_ep::expandRowsInPlace;
using hipdnn_ep::planConstantLayout;
using hipdnn_ep::rearrangeInPlace;
using hipdnn_ep::weightLayoutScratchBytes;

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
  EXPECT(planConstantLayout(64, 1000, 0, 0, 0, &l) == nullptr);
  EXPECT(l.offset == 64 && l.size == 1000 && l.src_size == 1000);
  EXPECT(l.rows == 0 && l.row_pitch == 0);
  EXPECT(planConstantLayout(0, 0, 0, 0, 0, &l) == nullptr);
}

// Expands `rows` rows of `row_bytes` (row r filled with value r + 1) to
// `row_pitch`, with guard bytes on both sides of the buffer.
static void checkExpansion(int64_t rows, int64_t row_bytes, int64_t row_pitch) {
  ConstantLayout l;
  EXPECT(planConstantLayout(0, rows * row_pitch, row_bytes, row_pitch, 0, &l) ==
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
  EXPECT(planConstantLayout(-1, 64, 0, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, -64, 0, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(kMax - 8, 64, 0, 0, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, -16, 64, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, 16, -64, 0, &l) != nullptr);
  // One row field set without the other.
  EXPECT(planConstantLayout(0, 128, 0, 64, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, 128, 64, 0, 0, &l) != nullptr);
  // Pitch smaller than the row.
  EXPECT(planConstantLayout(0, 128, 64, 32, 0, &l) != nullptr);
  // Footprint not a whole number of rows (would leave the tail unwritten),
  // including a pitch larger than the footprint.
  EXPECT(planConstantLayout(0, 130, 48, 64, 0, &l) != nullptr);
  EXPECT(planConstantLayout(0, 32, 16, 64, 0, &l) != nullptr);
}

// Where byte b of dense row n lands, from the layout definitions
// (include/hip/matmul_nbits_weight_layout.h), written out independently.
static uint64_t expectedOffset(int layout, uint64_t n, uint64_t b,
                               uint64_t rowBytes) {
  if (layout == 1) {
    if (rowBytes % 128 != 0)
      return n * rowBytes + b;
    const uint64_t lines = rowBytes / 128;
    uint64_t p = 1;
    while (lines % (p * 2) == 0)
      p *= 2;
    uint64_t g = 1; // gcd(lines, 64)
    for (uint64_t d = 64; d > 1; d /= 2)
      if (lines % d == 0) {
        g = d;
        break;
      }
    const uint64_t f = (n / (64 / g)) % p;
    return n * rowBytes + (((b / 128) ^ f) * 128) + b % 128;
  }
  if (layout == 2)
    return (n / 16) * 16 * rowBytes + (b / 32) * 512 + (n % 16) * 32 + b % 32;
  return n * rowBytes + b;
}

static void checkRearrange(int layout, int64_t rows, int64_t rowBytes) {
  ConstantLayout l;
  EXPECT(planConstantLayout(0, rows * rowBytes, rowBytes, rowBytes, layout,
                            &l) == nullptr);
  std::vector<uint8_t> src(static_cast<size_t>(rows * rowBytes));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>((i * 131 + i / 251) & 0xFF);
  std::vector<uint8_t> buf = src;
  std::vector<uint8_t> scratch(weightLayoutScratchBytes(l));
  rearrangeInPlace(buf.data(), l, scratch.data());
  for (uint64_t n = 0; n < static_cast<uint64_t>(rows); ++n)
    for (uint64_t b = 0; b < static_cast<uint64_t>(rowBytes); ++b)
      if (buf[expectedOffset(layout, n, b, rowBytes)] !=
          src[n * rowBytes + b]) {
        std::fprintf(stderr,
                     "layout %d rows %lld row %llu byte %llu misplaced\n",
                     layout, (long long)rows, (unsigned long long)n,
                     (unsigned long long)b);
        ++g_failures;
        return;
      }
}

static void testWeightLayouts() {
  // XorLines: K = 4096 (16 lines, P = 16, R = 4), K = 12288 (48 lines, P = 16),
  // K = 2880 (not whole lines: identity).
  checkRearrange(1, 64, 2048);
  checkRearrange(1, 40, 6144);
  checkRearrange(1, 8, 1440);
  // Blocks: 16-row groups of 64-value pieces.
  checkRearrange(2, 32, 2048);
  checkRearrange(2, 16, 32);
  ConstantLayout l;
  // Unknown codes, a pad, no rows, or a shape the layout is not defined for.
  EXPECT(planConstantLayout(0, 4096, 2048, 2048, 3, &l) != nullptr);
  EXPECT(planConstantLayout(0, 4096, 2048, 2048, -1, &l) != nullptr);
  EXPECT(planConstantLayout(0, 2 * 2304, 2048, 2304, 2, &l) != nullptr);
  EXPECT(planConstantLayout(0, 4096, 0, 0, 1, &l) != nullptr);
  EXPECT(planConstantLayout(0, 8 * 2048, 2048, 2048, 2, &l) != nullptr);
  EXPECT(planConstantLayout(0, 16 * 48, 48, 48, 2, &l) != nullptr);
  EXPECT(planConstantLayout(0, 16 * 2048, 2048, 2048, 2, &l) == nullptr);
  EXPECT(weightLayoutScratchBytes(l) == 16 * 2048);
}

int main() {
  testDense();
  testRowPadded();
  testMalformed();
  testWeightLayouts();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("constant layout: all checks passed\n");
  return 0;
}
