#include "mm_config.h"
#include "mm_hal.h"
#include "mm_kv.h"
#include <gtest/gtest.h>
#include <vector>

class KvBlockTest : public ::testing::Test {
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

TEST_F(KvBlockTest, AllocReturnsValidBlock) {
    mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
    ASSERT_NE(blk, MM_INVALID_BLOCK);

    void *ptr = mm_kv_get_block_ptr(blk);
    EXPECT_NE(ptr, nullptr);

    mm_kv_free_block(blk, 0);
}

TEST_F(KvBlockTest, FreeAndRecycle) {
    mm_kv_block_t blk1 = mm_kv_alloc_block(make_desc(), 0);
    ASSERT_NE(blk1, MM_INVALID_BLOCK);

    uint32_t free_before = mm_kv_free_block_count(MM_KV_FMT_FP16);
    mm_kv_free_block(blk1, 0);
    uint32_t free_after = mm_kv_free_block_count(MM_KV_FMT_FP16);
    EXPECT_EQ(free_after, free_before + 1);

    /* Re-alloc should succeed (recycled block) */
    mm_kv_block_t blk2 = mm_kv_alloc_block(make_desc(), 0);
    EXPECT_NE(blk2, MM_INVALID_BLOCK);

    mm_kv_free_block(blk2, 0);
}

TEST_F(KvBlockTest, FormatQueryMatchesDesc) {
    mm_kv_block_desc_t desc = make_desc();
    mm_kv_block_t blk = mm_kv_alloc_block(desc, 0);
    ASSERT_NE(blk, MM_INVALID_BLOCK);

    mm_kv_block_desc_t queried = mm_kv_get_format(blk);
    EXPECT_EQ(queried.format, desc.format);
    EXPECT_EQ(queried.num_kv_heads, desc.num_kv_heads);
    EXPECT_EQ(queried.head_dim, desc.head_dim);

    mm_kv_free_block(blk, 0);
}

TEST_F(KvBlockTest, BlockTablePointersDistinct) {
    constexpr int N = 5;
    mm_kv_block_t blocks[N];
    for (int i = 0; i < N; ++i) {
        blocks[i] = mm_kv_alloc_block(make_desc(), 0);
        ASSERT_NE(blocks[i], MM_INVALID_BLOCK);
    }

    void **table = mm_kv_get_block_table(blocks, N, MM_DEVICE_GPU_0);
    ASSERT_NE(table, nullptr);

    /* All pointers should be distinct */
    for (int i = 0; i < N; ++i) {
        EXPECT_NE(table[i], nullptr);
        for (int j = i + 1; j < N; ++j) {
            EXPECT_NE(table[i], table[j]);
        }
    }

    free(table);
    for (int i = 0; i < N; ++i)
        mm_kv_free_block(blocks[i], 0);
}

TEST_F(KvBlockTest, DoubleFreeIsSafe) {
    mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
    ASSERT_NE(blk, MM_INVALID_BLOCK);

    mm_kv_free_block(blk, 0);
    mm_kv_free_block(blk, 0); /* Should not crash */
}

TEST_F(KvBlockTest, AllocExhaustsPool) {
    std::vector<mm_kv_block_t> allocated;
    uint32_t total = mm_kv_total_block_count(MM_KV_FMT_FP16);

    /* Allocate all blocks */
    for (uint32_t i = 0; i < total; ++i) {
        mm_kv_block_t blk = mm_kv_alloc_block(make_desc(), 0);
        if (blk == MM_INVALID_BLOCK)
            break;
        allocated.push_back(blk);
    }

    /* Pool should be exhausted */
    EXPECT_EQ(mm_kv_free_block_count(MM_KV_FMT_FP16), 0u);

    /* Next alloc should fail */
    mm_kv_block_t fail = mm_kv_alloc_block(make_desc(), 0);
    EXPECT_EQ(fail, MM_INVALID_BLOCK);

    /* Free all */
    for (auto blk : allocated)
        mm_kv_free_block(blk, 0);
}

TEST_F(KvBlockTest, InvalidBlockOperations) {
    EXPECT_EQ(mm_kv_get_block_ptr(MM_INVALID_BLOCK), nullptr);

    mm_kv_block_desc_t desc = mm_kv_get_format(MM_INVALID_BLOCK);
    EXPECT_EQ(desc.format, (mm_kv_format_t)0);

    mm_kv_free_block(MM_INVALID_BLOCK, 0); /* Should not crash */
}
