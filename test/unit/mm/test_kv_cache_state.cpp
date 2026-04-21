/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_kv_cache_state.h"
#include <gtest/gtest.h>

class KvCacheStateTest : public ::testing::Test {
protected:
  void SetUp() override { state = mm_kv_cache_state_create(); }
  void TearDown() override {
    if (state)
      mm_kv_cache_state_destroy(state);
  }
  mm_kv_cache_state_t *state = nullptr;
};

TEST_F(KvCacheStateTest, CreateDestroy) {
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(mm_kv_cache_state_entry_count(state), 0u);
}

TEST_F(KvCacheStateTest, RegisterAndLookup) {
  char host_buf[1024];
  char gpu_buf[1024]; // Simulated GPU pointer

  int err = mm_kv_cache_state_register(state, host_buf, gpu_buf, 1024);
  EXPECT_EQ(err, MM_OK);
  EXPECT_EQ(mm_kv_cache_state_entry_count(state), 1u);

  auto *entry = mm_kv_cache_state_lookup(state, host_buf);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->host_ptr, (void *)host_buf);
  EXPECT_EQ(entry->gpu_ptr, (void *)gpu_buf);
  EXPECT_EQ(entry->size_bytes, 1024u);
  EXPECT_EQ(entry->inference_count, 0);
}

TEST_F(KvCacheStateTest, LookupMiss) {
  char buf[64];
  EXPECT_EQ(mm_kv_cache_state_lookup(state, buf), nullptr);
  EXPECT_EQ(mm_kv_cache_state_lookup(state, nullptr), nullptr);
}

TEST_F(KvCacheStateTest, NoDuplicateRegistration) {
  char host_buf[64];
  char gpu_buf1[64], gpu_buf2[64];

  mm_kv_cache_state_register(state, host_buf, gpu_buf1, 64);
  mm_kv_cache_state_register(state, host_buf, gpu_buf2, 128); // Duplicate

  EXPECT_EQ(mm_kv_cache_state_entry_count(state), 1u);

  // First registration wins
  auto *entry = mm_kv_cache_state_lookup(state, host_buf);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->gpu_ptr, (void *)gpu_buf1);
  EXPECT_EQ(entry->size_bytes, 64u);
}

TEST_F(KvCacheStateTest, MultipleEntries) {
  char host1[64], host2[64], host3[64];
  char gpu1[64], gpu2[64], gpu3[64];

  mm_kv_cache_state_register(state, host1, gpu1, 100);
  mm_kv_cache_state_register(state, host2, gpu2, 200);
  mm_kv_cache_state_register(state, host3, gpu3, 300);

  EXPECT_EQ(mm_kv_cache_state_entry_count(state), 3u);

  EXPECT_EQ(mm_kv_cache_state_lookup(state, host1)->size_bytes, 100u);
  EXPECT_EQ(mm_kv_cache_state_lookup(state, host2)->size_bytes, 200u);
  EXPECT_EQ(mm_kv_cache_state_lookup(state, host3)->size_bytes, 300u);
}

TEST_F(KvCacheStateTest, InferenceCountIncrement) {
  char host_buf[64], gpu_buf[64];
  mm_kv_cache_state_register(state, host_buf, gpu_buf, 64);

  auto *entry = mm_kv_cache_state_lookup(state, host_buf);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->inference_count, 0);

  entry->inference_count++;
  EXPECT_EQ(entry->inference_count, 1);

  entry->inference_count++;
  EXPECT_EQ(entry->inference_count, 2);

  // Verify the increment persists across lookups
  auto *entry2 = mm_kv_cache_state_lookup(state, host_buf);
  EXPECT_EQ(entry2->inference_count, 2);
}

TEST_F(KvCacheStateTest, NullStateSafe) {
  EXPECT_EQ(mm_kv_cache_state_lookup(nullptr, (void *)0x1), nullptr);
  EXPECT_EQ(mm_kv_cache_state_register(nullptr, (void *)0x1, (void *)0x2, 64),
            MM_ERROR_INVALID_ARG);
  EXPECT_EQ(mm_kv_cache_state_entry_count(nullptr), 0u);
  mm_kv_cache_state_destroy(nullptr); // Should not crash
}
