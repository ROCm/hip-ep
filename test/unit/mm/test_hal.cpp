#include "mm_hal.h"
#include <gtest/gtest.h>
#include <cstring>

class HalHostTest : public ::testing::Test {
protected:
    void SetUp() override { hal = mm_hal_host_get(); }
    const mm_hal_t *hal = nullptr;
};

TEST_F(HalHostTest, DeviceCountPositive) {
    EXPECT_GE(hal->get_device_count(), 1);
}

TEST_F(HalHostTest, DeviceInfoValid) {
    mm_device_info_t info = {};
    int err = hal->get_device_info(0, &info);
    EXPECT_EQ(err, MM_OK);
    EXPECT_GT(info.total_memory, 0u);
    EXPECT_GT(info.free_memory, 0u);
    EXPECT_GT(std::strlen(info.name), 0u);
}

TEST_F(HalHostTest, RawAllocFreeRoundTrip) {
    void *ptr = hal->raw_alloc(0, 4096, 64);
    ASSERT_NE(ptr, nullptr);

    /* Write pattern and verify */
    std::memset(ptr, 0xAB, 4096);
    EXPECT_EQ(static_cast<unsigned char *>(ptr)[0], 0xAB);
    EXPECT_EQ(static_cast<unsigned char *>(ptr)[4095], 0xAB);

    hal->raw_free(0, ptr);
}

TEST_F(HalHostTest, ZeroSizeAllocReturnsNull) {
    void *ptr = hal->raw_alloc(0, 0, 64);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(HalHostTest, AsyncCopyCorrectness) {
    char src[256], dst[256];
    std::memset(src, 0x42, sizeof(src));
    std::memset(dst, 0x00, sizeof(dst));

    int err = hal->async_copy(dst, src, sizeof(src),
                              MM_COPY_HOST_TO_HOST, 0, nullptr);
    EXPECT_EQ(err, MM_OK);
    EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);
}

TEST_F(HalHostTest, MemsetWorks) {
    void *ptr = hal->raw_alloc(0, 1024, 64);
    ASSERT_NE(ptr, nullptr);

    int err = hal->memset(ptr, 0xFF, 1024, 0);
    EXPECT_EQ(err, MM_OK);
    EXPECT_EQ(static_cast<unsigned char *>(ptr)[0], 0xFF);
    EXPECT_EQ(static_cast<unsigned char *>(ptr)[1023], 0xFF);

    hal->raw_free(0, ptr);
}

TEST_F(HalHostTest, HostAllocFree) {
    void *ptr = hal->host_alloc(8192, 64);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0xCD, 8192);
    EXPECT_EQ(static_cast<unsigned char *>(ptr)[0], 0xCD);
    hal->host_free(ptr);
}

TEST_F(HalHostTest, MemoryQueries) {
    size_t total = hal->get_total_memory(0);
    size_t free_mem = hal->get_free_memory(0);
    EXPECT_GT(total, 0u);
    EXPECT_GT(free_mem, 0u);
}

TEST_F(HalHostTest, RegisterAndGet) {
    int err = mm_hal_register(hal);
    EXPECT_EQ(err, MM_OK);
    EXPECT_EQ(mm_hal_get(), hal);
}

TEST_F(HalHostTest, RegisterNullFails) {
    int err = mm_hal_register(nullptr);
    EXPECT_EQ(err, MM_ERROR_INVALID_ARG);
}
