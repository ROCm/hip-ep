/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <gtest/gtest.h>

#include "mm/mm_api.h"
#include "mm/mm_kv.h"

namespace {

mm::Config make_config() {
  mm::Config cfg = mm::config_default();
  cfg.kv_cache_fraction = 0.10;
  cfg.kv_cache_max_bytes = 64 * 1024 * 1024; // 64 MB budget for tests.
  cfg.kv_block_size_tokens = 16;
  cfg.activation_slab_bytes = 128 * 1024;
  return cfg;
}

} // namespace

TEST(MemoryManagerTest, BasicAllocFree) {
  mm::Config cfg = make_config();
  ASSERT_EQ(mm::init(&cfg), mm::Status::Ok);

  mm::AllocHints hints;
  hints.mem_class = mm::MemoryClass::Generic;
  mm::handle_t handle = mm::alloc(4096, &hints);
  ASSERT_NE(handle, mm::kInvalidHandle);
  void *ptr = mm::get_ptr(handle);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(mm::free(handle), mm::Status::Ok);

  auto snap = mm::metrics_snapshot();
  EXPECT_EQ(snap.alloc_count, 1u);
  EXPECT_EQ(snap.free_count, 1u);

  mm::shutdown();
}

TEST(MemoryManagerTest, ActivationArenaReusesBlocks) {
  mm::Config cfg = make_config();
  cfg.kv_cache_fraction = 0.0; // dedicate memory to activation pool.
  cfg.kv_cache_max_bytes = 0;
  ASSERT_EQ(mm::init(&cfg), mm::Status::Ok);

  mm::AllocHints hints;
  hints.mem_class = mm::MemoryClass::Activation;
  hints.lifetime = mm::Lifetime::Step;

  mm::handle_t a = mm::alloc(2048, &hints);
  ASSERT_NE(a, mm::kInvalidHandle);
  void *ptr_a = mm::get_ptr(a);
  ASSERT_NE(ptr_a, nullptr);
  EXPECT_EQ(mm::free(a), mm::Status::Ok);

  mm::handle_t b = mm::alloc(2048, &hints);
  ASSERT_NE(b, mm::kInvalidHandle);
  void *ptr_b = mm::get_ptr(b);
  ASSERT_NE(ptr_b, nullptr);

  // Expect reuse from the arena free list.
  EXPECT_EQ(ptr_a, ptr_b);
  EXPECT_EQ(mm::free(b), mm::Status::Ok);

  mm::shutdown();
}

TEST(MemoryManagerTest, KvAllocForkAndFree) {
  mm::Config cfg = make_config();
  cfg.kv_cache_fraction = 0.0;
  cfg.kv_cache_max_bytes = 8 * 1024 * 1024;
  ASSERT_EQ(mm::init(&cfg), mm::Status::Ok);

  mm::KvBlockDesc desc;
  desc.block_size_tokens = 16;
  desc.bytes_per_token = 2048;
  desc.num_kv_heads = 1;
  desc.head_dim = 1;
  desc.num_layers = 1;

  mm::kv_block_t block = mm::kv_alloc_block(desc);
  ASSERT_NE(block, mm::kInvalidHandle);
  void *ptr_block = mm::get_ptr(block);
  ASSERT_NE(ptr_block, nullptr);

  mm::kv_block_t forked = mm::kv_fork_block(block);
  ASSERT_NE(forked, mm::kInvalidHandle);
  void *ptr_forked = mm::get_ptr(forked);
  EXPECT_EQ(ptr_block, ptr_forked);

  void *table[2] = {};
  mm::kv_block_t handles[2] = {block, forked};
  EXPECT_TRUE(mm::kv_get_block_table(handles, 2, table));
  EXPECT_EQ(table[0], ptr_block);
  EXPECT_EQ(table[1], ptr_block);

  EXPECT_EQ(mm::kv_free_block(forked), mm::Status::Ok);
  EXPECT_EQ(mm::kv_free_block(block), mm::Status::Ok);

  auto snap = mm::metrics_snapshot();
  EXPECT_GE(snap.kv_alloc_count, 2u);
  EXPECT_GE(snap.kv_free_count, 2u);

  mm::shutdown();
}
