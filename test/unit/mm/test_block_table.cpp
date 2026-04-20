#include "mm_block_table.h"
#include "mm_config.h"
#include "mm_hal.h"
#include "mm_kv.h"
#include <gtest/gtest.h>
#include <cstring>

class BlockTableTest : public ::testing::Test {
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

TEST_F(BlockTableTest, BuildTable) {
    mm::BlockTable table;
    EXPECT_EQ(table.size(), 0u);

    mm_kv_block_t blk1 = mm_kv_alloc_block(make_desc(), 0);
    mm_kv_block_t blk2 = mm_kv_alloc_block(make_desc(), 0);
    ASSERT_NE(blk1, MM_INVALID_BLOCK);
    ASSERT_NE(blk2, MM_INVALID_BLOCK);

    table.add_block(mm_kv_get_block_ptr(blk1));
    table.add_block(mm_kv_get_block_ptr(blk2));

    EXPECT_EQ(table.size(), 2u);
    EXPECT_NE(table.data(), nullptr);
    EXPECT_NE(table.data()[0], nullptr);
    EXPECT_NE(table.data()[1], nullptr);
    EXPECT_NE(table.data()[0], table.data()[1]);

    mm_kv_free_block(blk1, 0);
    mm_kv_free_block(blk2, 0);
}

TEST_F(BlockTableTest, ClearTable) {
    mm::BlockTable table;

    mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
    ASSERT_NE(blk, MM_INVALID_BLOCK);

    table.add_block(mm_kv_get_block_ptr(blk));
    EXPECT_EQ(table.size(), 1u);

    table.clear();
    EXPECT_EQ(table.size(), 0u);

    mm_kv_free_block(blk, 0);
}

TEST_F(BlockTableTest, MaterializeContiguous) {
    /* Allocate 3 blocks and write distinct patterns */
    constexpr int N = 3;
    mm_kv_block_t blocks[N];
    for (int i = 0; i < N; ++i) {
        blocks[i] = mm_kv_alloc_block(make_desc(), 0);
        ASSERT_NE(blocks[i], MM_INVALID_BLOCK);
        /* Write pattern: block 0 = 0xAA, block 1 = 0xBB, block 2 = 0xCC */
        void *ptr = mm_kv_get_block_ptr(blocks[i]);
        ASSERT_NE(ptr, nullptr);
        std::memset(ptr, 0xAA + i * 0x11, 256);
    }

    /* Build block table */
    mm::BlockTable table;
    for (int i = 0; i < N; ++i)
        table.add_block(mm_kv_get_block_ptr(blocks[i]));

    /* Materialize into contiguous buffer */
    size_t block_size = 256;
    std::vector<unsigned char> dst(N * block_size, 0);
    const mm_hal_t *hal = mm_hal_host_get();
    int err = table.materialize_contiguous(dst.data(), block_size, hal, 0);
    EXPECT_EQ(err, MM_OK);

    /* Verify patterns are contiguous */
    EXPECT_EQ(dst[0], 0xAA);
    EXPECT_EQ(dst[block_size], 0xBB);
    EXPECT_EQ(dst[2 * block_size], 0xCC);

    for (int i = 0; i < N; ++i)
        mm_kv_free_block(blocks[i], 0);
}

TEST_F(BlockTableTest, GetBlockTableAPI) {
    constexpr int N = 4;
    mm_kv_block_t blocks[N];
    for (int i = 0; i < N; ++i) {
        blocks[i] = mm_kv_alloc_block(make_desc(), 0);
        ASSERT_NE(blocks[i], MM_INVALID_BLOCK);
    }

    void **table = mm_kv_get_block_table(blocks, N, MM_DEVICE_GPU_0);
    ASSERT_NE(table, nullptr);

    for (int i = 0; i < N; ++i)
        EXPECT_NE(table[i], nullptr);

    free(table);
    for (int i = 0; i < N; ++i)
        mm_kv_free_block(blocks[i], 0);
}
