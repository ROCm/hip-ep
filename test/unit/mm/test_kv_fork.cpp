#include "mm_config.h"
#include "mm_hal.h"
#include "mm_kv.h"
#include <gtest/gtest.h>

class KvForkTest : public ::testing::Test {
protected:
  void SetUp() override {
    mm_hal_register(mm_hal_host_get());
    mm_config_t config = {};
    config.gpu_memory_limit = 64 * 1024 * 1024;
    config.kv_cache_fraction = 0.5f;
    config.kv_block_size_tokens = 16;
    config.num_size_classes = 4;
    config.high_watermark = 0.90f;
    config.critical_watermark = 0.95f;
    config.default_kv_format = MM_KV_FMT_FP16;
    ASSERT_EQ(mm_init(&config), MM_OK);
  }
  void TearDown() override { mm_shutdown(); }

  mm_kv_block_desc_t make_desc() {
    mm_kv_block_desc_t desc = {};
    desc.format = MM_KV_FMT_FP16;
    desc.block_size_tokens = 16;
    desc.num_kv_heads = 4;
    desc.head_dim = 128;
    return desc;
  }
};

TEST_F(KvForkTest, ForkSharesPhysicalMemory) {
  mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
  ASSERT_NE(blk, MM_INVALID_BLOCK);

  void *ptr_orig = mm_kv_get_block_ptr(blk);
  ASSERT_NE(ptr_orig, nullptr);

  mm_kv_block_t forked = mm_kv_fork_block(blk);
  ASSERT_NE(forked, MM_INVALID_BLOCK);

  /* Fork shares same physical memory (CoW) */
  void *ptr_forked = mm_kv_get_block_ptr(forked);
  EXPECT_EQ(ptr_orig, ptr_forked);

  mm_kv_free_block(blk, 0);
  mm_kv_free_block(forked, 0);
}

TEST_F(KvForkTest, ForkSurvivesOriginalFree) {
  mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
  ASSERT_NE(blk, MM_INVALID_BLOCK);

  mm_kv_block_t forked = mm_kv_fork_block(blk);
  ASSERT_NE(forked, MM_INVALID_BLOCK);

  uint32_t free_before = mm_kv_free_block_count(MM_KV_FMT_FP16);

  /* Free the original — forked still holds a reference */
  mm_kv_free_block(blk, 0);

  /* Block should NOT be returned to free list (forked still alive) */
  uint32_t free_after = mm_kv_free_block_count(MM_KV_FMT_FP16);
  EXPECT_EQ(free_before, free_after);

  /* Forked block should still be valid */
  void *ptr = mm_kv_get_block_ptr(forked);
  EXPECT_NE(ptr, nullptr);

  /* Now free forked — block goes back to free list */
  mm_kv_free_block(forked, 0);
  uint32_t free_final = mm_kv_free_block_count(MM_KV_FMT_FP16);
  EXPECT_EQ(free_final, free_before + 1);
}

TEST_F(KvForkTest, MultipleForks) {
  mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
  ASSERT_NE(blk, MM_INVALID_BLOCK);

  /* Fork 5 times */
  mm_kv_block_t forks[5];
  for (int i = 0; i < 5; ++i) {
    forks[i] = mm_kv_fork_block(blk);
    ASSERT_NE(forks[i], MM_INVALID_BLOCK);
  }

  /* All share same pointer */
  void *orig_ptr = mm_kv_get_block_ptr(blk);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(mm_kv_get_block_ptr(forks[i]), orig_ptr);
  }

  /* Free original + 4 forks — block should NOT return to free list */
  uint32_t free_before = mm_kv_free_block_count(MM_KV_FMT_FP16);
  mm_kv_free_block(blk, 0);
  for (int i = 0; i < 4; ++i)
    mm_kv_free_block(forks[i], 0);
  EXPECT_EQ(mm_kv_free_block_count(MM_KV_FMT_FP16), free_before);

  /* Free last fork — now block returns */
  mm_kv_free_block(forks[4], 0);
  EXPECT_EQ(mm_kv_free_block_count(MM_KV_FMT_FP16), free_before + 1);
}

TEST_F(KvForkTest, ForkInvalidBlockFails) {
  mm_kv_block_t forked = mm_kv_fork_block(MM_INVALID_BLOCK);
  EXPECT_EQ(forked, MM_INVALID_BLOCK);
}
