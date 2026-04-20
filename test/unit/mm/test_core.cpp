#include "mm_config.h"
#include "mm_core.h"
#include "mm_hal.h"
#include <cstring>
#include <gtest/gtest.h>

class CoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    mm_hal_register(mm_hal_host_get());
    mm_config_t config = {};
    config.gpu_memory_limit = 64 * 1024 * 1024;
    config.kv_cache_fraction = 0.5f;
    config.num_size_classes = 8;
    config.high_watermark = 0.90f;
    config.critical_watermark = 0.95f;
    ASSERT_EQ(mm_init(&config), MM_OK);
  }
  void TearDown() override { mm_shutdown(); }
};

TEST_F(CoreTest, AllocFreeActivation) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_ACTIVATION;
  hints.lifetime = MM_LIFETIME_STEP;

  mm_handle_t h = mm_alloc(4096, hints, 0);
  ASSERT_NE(h, MM_INVALID_HANDLE);

  void *ptr = mm_get_ptr(h, MM_DEVICE_GPU_0);
  ASSERT_NE(ptr, nullptr);

  /* Write and verify */
  std::memset(ptr, 0xCD, 4096);
  EXPECT_EQ(static_cast<unsigned char *>(ptr)[0], 0xCD);

  mm_free(h, 0);
}

TEST_F(CoreTest, AllocFreeScratch) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_SCRATCH;
  hints.lifetime = MM_LIFETIME_TRANSIENT;

  mm_handle_t h = mm_alloc(2048, hints, 0);
  ASSERT_NE(h, MM_INVALID_HANDLE);

  void *ptr = mm_get_ptr(h, MM_DEVICE_GPU_0);
  ASSERT_NE(ptr, nullptr);

  mm_free(h, 0);
}

TEST_F(CoreTest, AllocWeightFails) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_WEIGHT;
  mm_handle_t h = mm_alloc(1024, hints, 0);
  EXPECT_EQ(h, MM_INVALID_HANDLE);
}

TEST_F(CoreTest, AllocKvCacheFails) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_KV_CACHE;
  mm_handle_t h = mm_alloc(1024, hints, 0);
  EXPECT_EQ(h, MM_INVALID_HANDLE);
}

TEST_F(CoreTest, QueryInfo) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_ACTIVATION;
  hints.lifetime = MM_LIFETIME_STEP;

  mm_handle_t h = mm_alloc(8192, hints, 0);
  ASSERT_NE(h, MM_INVALID_HANDLE);

  mm_alloc_info_t info = mm_query(h);
  EXPECT_EQ(info.mem_class, MM_CLASS_ACTIVATION);
  EXPECT_EQ(info.size, 8192u);
  EXPECT_EQ(info.ref_count, 1u);
  EXPECT_EQ(info.current_tier, MM_TIER_HBM);

  mm_free(h, 0);
}

TEST_F(CoreTest, GetPtrAfterFreeReturnsNull) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_ACTIVATION;

  mm_handle_t h = mm_alloc(1024, hints, 0);
  ASSERT_NE(h, MM_INVALID_HANDLE);

  mm_free(h, 0);
  EXPECT_EQ(mm_get_ptr(h, MM_DEVICE_GPU_0), nullptr);
}

TEST_F(CoreTest, InvalidHandleGetPtrReturnsNull) {
  EXPECT_EQ(mm_get_ptr(MM_INVALID_HANDLE, MM_DEVICE_GPU_0), nullptr);
  EXPECT_EQ(mm_get_ptr(9999, MM_DEVICE_GPU_0), nullptr);
}

TEST_F(CoreTest, LargeAllocBfcFallback) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_SCRATCH;
  hints.lifetime = MM_LIFETIME_REQUEST;

  /* 8 MB should go through BFC */
  mm_handle_t h = mm_alloc(8 * 1024 * 1024, hints, 0);
  ASSERT_NE(h, MM_INVALID_HANDLE);

  void *ptr = mm_get_ptr(h, MM_DEVICE_GPU_0);
  ASSERT_NE(ptr, nullptr);

  mm_free(h, 0);
}

TEST_F(CoreTest, MultipleAllocsDifferentPointers) {
  mm_alloc_hints_t hints = {};
  hints.mem_class = MM_CLASS_ACTIVATION;

  mm_handle_t h1 = mm_alloc(1024, hints, 0);
  mm_handle_t h2 = mm_alloc(1024, hints, 0);
  ASSERT_NE(h1, MM_INVALID_HANDLE);
  ASSERT_NE(h2, MM_INVALID_HANDLE);
  EXPECT_NE(h1, h2);

  void *p1 = mm_get_ptr(h1, MM_DEVICE_GPU_0);
  void *p2 = mm_get_ptr(h2, MM_DEVICE_GPU_0);
  EXPECT_NE(p1, p2);

  mm_free(h1, 0);
  mm_free(h2, 0);
}
