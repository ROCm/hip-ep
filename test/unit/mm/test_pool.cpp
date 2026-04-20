#include "mm_config.h"
#include "mm_hal.h"
#include "mm_pool.h"
#include <gtest/gtest.h>
#include <cstring>

class PoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        mm_hal_register(mm_hal_host_get());
        mm_config_t config = {};
        config.gpu_memory_limit = 64 * 1024 * 1024;
        config.kv_cache_fraction = 0.5f;
        config.num_size_classes = 4;
        config.high_watermark = 0.90f;
        config.critical_watermark = 0.95f;
        ASSERT_EQ(mm_init(&config), MM_OK);
    }
    void TearDown() override { mm_shutdown(); }
};

TEST_F(PoolTest, CreateAndLookupTensors) {
    mm_buffer_entry_t entries[] = {
        {0, 0,    4096, 256},   /* tensor 0 at offset 0 */
        {1, 4096, 2048, 256},   /* tensor 1 at offset 4096 */
        {2, 8192, 1024, 256},   /* tensor 2 at offset 8192 */
    };

    mm_static_plan_t plan = {};
    plan.total_size = 16384;
    plan.mem_class = MM_CLASS_WEIGHT;
    plan.device = MM_DEVICE_CPU;
    plan.num_entries = 3;
    plan.entries = entries;

    mm_pool_t pool = mm_create_pool(&plan);
    ASSERT_NE(pool, MM_INVALID_POOL);

    void *p0 = mm_pool_get_ptr(pool, 0);
    void *p1 = mm_pool_get_ptr(pool, 1);
    void *p2 = mm_pool_get_ptr(pool, 2);

    ASSERT_NE(p0, nullptr);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);

    /* Verify offsets are correct (p1 = p0 + 4096, p2 = p0 + 8192) */
    EXPECT_EQ(static_cast<char *>(p1) - static_cast<char *>(p0), 4096);
    EXPECT_EQ(static_cast<char *>(p2) - static_cast<char *>(p0), 8192);

    mm_destroy_pool(pool);
}

TEST_F(PoolTest, InvalidTensorIdReturnsNull) {
    mm_buffer_entry_t entries[] = {{0, 0, 1024, 256}};
    mm_static_plan_t plan = {};
    plan.total_size = 1024;
    plan.mem_class = MM_CLASS_WEIGHT;
    plan.device = MM_DEVICE_CPU;
    plan.num_entries = 1;
    plan.entries = entries;

    mm_pool_t pool = mm_create_pool(&plan);
    ASSERT_NE(pool, MM_INVALID_POOL);

    EXPECT_EQ(mm_pool_get_ptr(pool, 99), nullptr);

    mm_destroy_pool(pool);
}

TEST_F(PoolTest, DestroyReleasesMemory) {
    mm_buffer_entry_t entries[] = {{0, 0, 1024, 256}};
    mm_static_plan_t plan = {};
    plan.total_size = 1024;
    plan.mem_class = MM_CLASS_WEIGHT;
    plan.device = MM_DEVICE_CPU;
    plan.num_entries = 1;
    plan.entries = entries;

    mm_pool_t pool = mm_create_pool(&plan);
    ASSERT_NE(pool, MM_INVALID_POOL);

    mm_destroy_pool(pool);
    /* After destroy, get_ptr should return nullptr */
    EXPECT_EQ(mm_pool_get_ptr(pool, 0), nullptr);
}

TEST_F(PoolTest, NullPlanFails) {
    EXPECT_EQ(mm_create_pool(nullptr), MM_INVALID_POOL);
}

TEST_F(PoolTest, WriteAndReadData) {
    mm_buffer_entry_t entries[] = {
        {0, 0,    256, 256},
        {1, 256,  256, 256},
    };
    mm_static_plan_t plan = {};
    plan.total_size = 512;
    plan.mem_class = MM_CLASS_WEIGHT;
    plan.device = MM_DEVICE_CPU;
    plan.num_entries = 2;
    plan.entries = entries;

    mm_pool_t pool = mm_create_pool(&plan);
    ASSERT_NE(pool, MM_INVALID_POOL);

    /* Write distinct patterns to each tensor */
    std::memset(mm_pool_get_ptr(pool, 0), 0xAA, 256);
    std::memset(mm_pool_get_ptr(pool, 1), 0xBB, 256);

    /* Verify patterns are distinct and correct */
    auto *t0 = static_cast<unsigned char *>(mm_pool_get_ptr(pool, 0));
    auto *t1 = static_cast<unsigned char *>(mm_pool_get_ptr(pool, 1));
    EXPECT_EQ(t0[0], 0xAA);
    EXPECT_EQ(t0[255], 0xAA);
    EXPECT_EQ(t1[0], 0xBB);
    EXPECT_EQ(t1[255], 0xBB);

    mm_destroy_pool(pool);
}
