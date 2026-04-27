/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * test_hal_mock — Exercise every function in the mock HAL vtable.
 */

#include "mm/mm_hal.h"
#include <cassert>
#include <cstdio>
#include <cstring>

#define ASSERT_OK(expr) do { mm_status_t s_ = (expr); assert(s_ == MM_OK); (void)s_; } while(0)
#define ASSERT_ERR(expr, code) do { mm_status_t s_ = (expr); assert(s_ == (code)); (void)s_; } while(0)

int main() {
    const mm_hal_t* hal = mm_hal_mock();
    assert(hal != nullptr);

    /* malloc / free */
    void* ptr = nullptr;
    ASSERT_OK(hal->malloc(&ptr, 1024));
    assert(ptr != nullptr);
    ASSERT_OK(hal->free(ptr));

    /* malloc zero size => error */
    void* bad = nullptr;
    ASSERT_ERR(hal->malloc(&bad, 0), MM_ERR_INVALID_ARGUMENT);

    /* memcpy round-trip */
    void* gpu = nullptr;
    ASSERT_OK(hal->malloc(&gpu, 256));

    unsigned char src[256];
    for (int i = 0; i < 256; i++)
        src[i] = (unsigned char)i;

    ASSERT_OK(hal->memcpy_h2d(gpu, src, 256, nullptr));

    unsigned char dst[256] = {};
    ASSERT_OK(hal->memcpy_d2h(dst, gpu, 256, nullptr));
    assert(memcmp(src, dst, 256) == 0);

    /* memset */
    ASSERT_OK(hal->memset(gpu, 0xAB, 256, nullptr));
    unsigned char check[256] = {};
    ASSERT_OK(hal->memcpy_d2h(check, gpu, 256, nullptr));
    for (int i = 0; i < 256; i++)
        assert(check[i] == 0xAB);

    ASSERT_OK(hal->free(gpu));

    /* stream create / sync / destroy */
    mm_stream_t stream = nullptr;
    ASSERT_OK(hal->stream_create(&stream));
    assert(stream != nullptr);
    ASSERT_OK(hal->stream_sync(stream));
    ASSERT_OK(hal->stream_destroy(stream));

    /* device queries */
    size_t free_bytes = 0, total_bytes = 0;
    ASSERT_OK(hal->get_free_mem(&free_bytes, &total_bytes));
    assert(total_bytes > 0);

    ASSERT_OK(hal->set_device(0));

    printf("test_hal_mock: PASSED\n");
    return 0;
}
