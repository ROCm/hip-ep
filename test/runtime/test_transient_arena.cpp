/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* GPU-free tests for the session transient arena's offset allocator
 * (lib/Runtime/transient_arena.h): alignment, first-fit reuse, coalescing,
 * exhaustion accounting and statistics.
 */
#include "transient_arena.h"

#include <cstdio>

using hipdnn_ep::TransientArenaAllocator;

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

constexpr size_t kAlign = TransientArenaAllocator::kAlign;

static void testRoundUp() {
  CHECK(TransientArenaAllocator::roundUp(0) == kAlign, "zero rounds to kAlign");
  CHECK(TransientArenaAllocator::roundUp(1) == kAlign, "1 rounds to kAlign");
  CHECK(TransientArenaAllocator::roundUp(kAlign) == kAlign, "exact multiple");
  CHECK(TransientArenaAllocator::roundUp(kAlign + 1) == 2 * kAlign,
        "rounds up to next multiple");
}

static void testInitTruncatesToAlignment() {
  TransientArenaAllocator a;
  a.init(10 * kAlign + 123);
  CHECK(a.size() == 10 * kAlign, "size truncated to kAlign multiple");
  CHECK(a.freeBlockCount() == 1, "one free block after init");

  TransientArenaAllocator empty;
  empty.init(kAlign - 1);
  CHECK(empty.size() == 0, "sub-kAlign arena is empty");
  CHECK(!empty.allocate(1), "empty arena cannot allocate");
  CHECK(empty.misses() == 1, "empty-arena request counted as a miss");
}

static void testAlignmentAndFirstFit() {
  TransientArenaAllocator a;
  a.init(16 * kAlign);
  auto x = a.allocate(1);
  auto y = a.allocate(kAlign + 1);
  auto z = a.allocate(kAlign);
  CHECK(x && *x == 0, "first block at offset 0");
  CHECK(y && *y == kAlign, "second block follows the first");
  CHECK(z && *z == 3 * kAlign, "third block follows the rounded second");
  CHECK(a.blockBytes(*y) == 2 * kAlign, "block bytes are rounded");
  CHECK(*x % kAlign == 0 && *y % kAlign == 0 && *z % kAlign == 0,
        "offsets are kAlign aligned");

  // Freeing the middle block leaves a 2-page hole; a 1-page request takes it
  // (first fit) rather than the tail.
  CHECK(a.release(*y), "release middle");
  auto w = a.allocate(kAlign);
  CHECK(w && *w == kAlign, "first fit reuses the lowest hole");
  // A 2-page request skips the remaining 1-page hole.
  auto v = a.allocate(2 * kAlign);
  CHECK(v && *v == 4 * kAlign, "too-small hole is skipped");
}

static void testCoalescing() {
  TransientArenaAllocator a;
  a.init(4 * kAlign);
  auto b0 = a.allocate(kAlign);
  auto b1 = a.allocate(kAlign);
  auto b2 = a.allocate(kAlign);
  auto b3 = a.allocate(kAlign);
  CHECK(b0 && b1 && b2 && b3, "fill arena");
  CHECK(a.freeBlockCount() == 0, "full arena has no free blocks");

  CHECK(a.release(*b1), "release b1");
  CHECK(a.release(*b0), "release b0 (merges with next)");
  CHECK(a.freeBlockCount() == 1, "b0+b1 coalesced with next");

  CHECK(a.release(*b3), "release b3");
  CHECK(a.freeBlockCount() == 2, "b3 not adjacent to b0+b1");
  CHECK(a.release(*b2), "release b2 (merges with prev and next)");
  CHECK(a.freeBlockCount() == 1, "all blocks coalesced");
  CHECK(a.used() == 0, "nothing used after full release");

  auto whole = a.allocate(4 * kAlign);
  CHECK(whole && *whole == 0, "coalesced arena serves a full-size request");

  TransientArenaAllocator p;
  p.init(3 * kAlign);
  auto p0 = p.allocate(kAlign);
  auto p1 = p.allocate(kAlign);
  CHECK(p0 && p1, "fill prefix");
  CHECK(p.release(*p0), "release p0");
  CHECK(p.release(*p1), "release p1 (merges with prev and the tail)");
  CHECK(p.freeBlockCount() == 1, "prev-merge coalesced");
}

static void testExhaustionAndStats() {
  TransientArenaAllocator a;
  a.init(4 * kAlign);
  auto x = a.allocate(3 * kAlign);
  CHECK(x, "fits");
  CHECK(!a.allocate(2 * kAlign), "does not fit in the remaining page");
  CHECK(!a.allocate(5 * kAlign), "larger than the arena");
  CHECK(a.hits() == 1, "one hit");
  CHECK(a.misses() == 2, "two misses");
  CHECK(a.missBytes() == 7 * kAlign, "miss bytes are the requested sizes");
  CHECK(a.used() == 3 * kAlign, "used after one allocation");

  auto y = a.allocate(kAlign);
  CHECK(y, "last page fits");
  CHECK(a.highWaterMark() == 4 * kAlign, "high-water mark at full");
  CHECK(a.release(*x) && a.release(*y), "release all");
  CHECK(a.used() == 0, "used back to zero");
  CHECK(a.highWaterMark() == 4 * kAlign, "high-water mark is sticky");
}

static void testUnknownRelease() {
  TransientArenaAllocator a;
  a.init(4 * kAlign);
  auto x = a.allocate(kAlign);
  CHECK(x, "allocate");
  CHECK(!a.release(*x + kAlign), "offset that is not a block start");
  CHECK(!a.release(100 * kAlign), "offset outside the arena");
  CHECK(a.blockBytes(*x + kAlign) == 0, "no bytes for unknown offset");
  CHECK(a.release(*x), "release live block");
  CHECK(!a.release(*x), "double release rejected");
  CHECK(a.used() == 0, "rejected releases do not change accounting");
  CHECK(a.freeBlockCount() == 1, "rejected releases do not split free list");
}

int main() {
  testRoundUp();
  testInitTruncatesToAlignment();
  testAlignmentAndFirstFit();
  testCoalescing();
  testExhaustionAndStats();
  testUnknownRelease();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("All transient arena tests passed\n");
  return 0;
}
