/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_handle_table.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

TEST(HandleTableTest, InsertAndLookup) {
  mm::HandleTable table;
  int dummy = 42;
  mm_handle_t h = table.insert(&dummy, sizeof(dummy), MM_CLASS_ACTIVATION);
  EXPECT_NE(h, MM_INVALID_HANDLE);

  auto *entry = table.lookup(h);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->ptr, &dummy);
  EXPECT_EQ(entry->size, sizeof(dummy));
  EXPECT_EQ(entry->mem_class, MM_CLASS_ACTIVATION);
  EXPECT_TRUE(entry->active);
}

TEST(HandleTableTest, MonotonicIds) {
  mm::HandleTable table;
  int a, b, c;
  mm_handle_t h1 = table.insert(&a, 1, MM_CLASS_SCRATCH);
  mm_handle_t h2 = table.insert(&b, 2, MM_CLASS_SCRATCH);
  mm_handle_t h3 = table.insert(&c, 3, MM_CLASS_SCRATCH);
  EXPECT_LT(h1, h2);
  EXPECT_LT(h2, h3);
}

TEST(HandleTableTest, RemoveInvalidatesHandle) {
  mm::HandleTable table;
  int dummy = 0;
  mm_handle_t h = table.insert(&dummy, sizeof(dummy), MM_CLASS_ACTIVATION);

  auto *removed = table.remove(h);
  ASSERT_NE(removed, nullptr);
  EXPECT_EQ(removed->ptr, &dummy);

  /* Lookup after remove returns nullptr */
  EXPECT_EQ(table.lookup(h), nullptr);
}

TEST(HandleTableTest, LookupInvalidHandle) {
  mm::HandleTable table;
  EXPECT_EQ(table.lookup(MM_INVALID_HANDLE), nullptr);
  EXPECT_EQ(table.lookup(999), nullptr);
}

TEST(HandleTableTest, ActiveCount) {
  mm::HandleTable table;
  int a, b;
  mm_handle_t h1 = table.insert(&a, 1, MM_CLASS_ACTIVATION);
  table.insert(&b, 2, MM_CLASS_SCRATCH);
  EXPECT_EQ(table.active_count(), 2u);

  table.remove(h1);
  EXPECT_EQ(table.active_count(), 1u);
}

TEST(HandleTableTest, StressInsertRemove) {
  mm::HandleTable table;
  constexpr int N = 10000;
  std::vector<mm_handle_t> handles(N);
  int data[1];

  for (int i = 0; i < N; ++i)
    handles[i] = table.insert(data, sizeof(data), MM_CLASS_ACTIVATION);

  EXPECT_EQ(table.active_count(), (size_t)N);

  for (int i = 0; i < N; ++i)
    table.remove(handles[i]);

  EXPECT_EQ(table.active_count(), 0u);
}
