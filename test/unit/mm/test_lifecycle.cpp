#include "mm_config.h"
#include "mm_core.h"
#include "mm_hal.h"
#include <gtest/gtest.h>

class LifecycleTest : public ::testing::Test {
protected:
  void SetUp() override { mm_hal_register(mm_hal_host_get()); }
  void TearDown() override { mm_shutdown(); }
};

TEST_F(LifecycleTest, InitWithDefaults) {
  int err = mm_init(nullptr);
  EXPECT_EQ(err, MM_OK);
}

TEST_F(LifecycleTest, InitWithConfig) {
  mm_config_t config = {};
  config.gpu_memory_limit = 64 * 1024 * 1024; /* 64 MB */
  config.kv_cache_fraction = 0.50f;
  config.kv_block_size_tokens = 16;
  config.max_tier = MM_TIER_DRAM;
  config.enable_prefix_caching = true;
  config.enable_defrag = false;
  config.num_size_classes = 8;
  config.high_watermark = 0.85f;
  config.critical_watermark = 0.95f;
  config.default_kv_format = MM_KV_FMT_FP16;
  config.pressure_kv_format = MM_KV_FMT_TURBOQUANT_4;
  config.critical_kv_format = MM_KV_FMT_TURBOQUANT_3;
  config.enable_inline_quant = false;
  config.enable_adaptive_transcode = true;

  int err = mm_init(&config);
  EXPECT_EQ(err, MM_OK);
}

TEST_F(LifecycleTest, DoubleInitFails) {
  ASSERT_EQ(mm_init(nullptr), MM_OK);
  EXPECT_EQ(mm_init(nullptr), MM_ERROR_ALREADY_INIT);
}

TEST_F(LifecycleTest, ShutdownWithoutInitIsSafe) {
  mm_shutdown(); /* Should not crash */
}

TEST_F(LifecycleTest, AllocBeforeInitFails) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_ACTIVATION;
  mm_handle_t h = mm_alloc(1024, hints, 0);
  EXPECT_EQ(h, MM_INVALID_HANDLE);
}

TEST_F(LifecycleTest, InvalidConfigRejected) {
  mm_config_t config = {};
  config.kv_cache_fraction = 1.5f; /* invalid: > 1.0 */
  config.num_size_classes = 8;
  config.high_watermark = 0.90f;
  config.critical_watermark = 0.95f;
  EXPECT_EQ(mm_init(&config), MM_ERROR_INVALID_ARG);
}

TEST_F(LifecycleTest, ReinitAfterShutdown) {
  ASSERT_EQ(mm_init(nullptr), MM_OK);
  mm_shutdown();
  EXPECT_EQ(mm_init(nullptr), MM_OK);
}
