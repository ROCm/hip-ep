/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_arena.h"
#include "mm_hal.h"
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <vector>

class ArenaTest : public ::testing::Test {
protected:
  void SetUp() override {
    hal = mm_hal_host_get();
    ASSERT_EQ(arena.init(hal, 0, 16 * 1024 * 1024, 8), MM_OK);
  }
  void TearDown() override { arena.shutdown(); }

  const mm_hal_t *hal = nullptr;
  mm::ArenaAllocator arena;
};

TEST_F(ArenaTest, BumpAlloc) {
  void *p1 = arena.alloc(512);
  void *p2 = arena.alloc(512);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_NE(p1, p2);
}

TEST_F(ArenaTest, SizeClassRouting) {
  EXPECT_EQ(mm::ArenaAllocator::size_class(100), 0u);    /* <1KB → class 0 */
  EXPECT_EQ(mm::ArenaAllocator::size_class(1024), 1u);   /* 1KB → class 1 */
  EXPECT_EQ(mm::ArenaAllocator::size_class(4096), 2u);   /* 4KB → class 2 */
  EXPECT_EQ(mm::ArenaAllocator::size_class(100000), 4u); /* 100KB → class 4 */
  EXPECT_EQ(mm::ArenaAllocator::size_class(5000000),
            7u); /* 5MB → class 7 (BFC) */
}

TEST_F(ArenaTest, StepReset) {
  void *p1 = arena.alloc(256);
  ASSERT_NE(p1, nullptr);
  size_t used_before = arena.used_bytes();
  EXPECT_GT(used_before, 0u);

  arena.reset();
  /* After reset, arena bump pointers are back to 0 */
  /* (BFC allocations are not affected by reset) */

  void *p2 = arena.alloc(256);
  ASSERT_NE(p2, nullptr);
  /* p2 should reuse the same memory as p1 in the same class */
  EXPECT_EQ(p1, p2);
}

TEST_F(ArenaTest, BfcFallbackForLargeAlloc) {
  void *ptr = arena.alloc(8 * 1024 * 1024); /* 8 MB → BFC */
  ASSERT_NE(ptr, nullptr);
  arena.free(ptr, 8 * 1024 * 1024);
}

TEST_F(ArenaTest, ConcurrentBumpNoOverlap) {
  constexpr int kThreads = 4;
  constexpr int kAllocsPerThread = 50;
  constexpr size_t kAllocSize =
      256; /* Must be >= kDefaultAlignment to get unique bumps */

  std::vector<std::vector<void *>> results(kThreads);
  std::vector<std::thread> threads;

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([this, &results, t, kAllocsPerThread, kAllocSize]() {
      for (int i = 0; i < kAllocsPerThread; ++i) {
        void *p = arena.alloc(kAllocSize);
        ASSERT_NE(p, nullptr);
        results[t].push_back(p);
      }
    });
  }

  for (auto &th : threads)
    th.join();

  /* Verify no two allocations overlap */
  std::set<void *> all_ptrs;
  for (const auto &vec : results) {
    for (void *p : vec) {
      auto [_, inserted] = all_ptrs.insert(p);
      EXPECT_TRUE(inserted) << "Duplicate allocation pointer detected";
    }
  }
  EXPECT_EQ(all_ptrs.size(), (size_t)(kThreads * kAllocsPerThread));
}

TEST_F(ArenaTest, UsedBytesTracking) {
  size_t before = arena.used_bytes();
  arena.alloc(1024);
  size_t after = arena.used_bytes();
  EXPECT_GT(after, before);
}
