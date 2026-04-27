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

int main() {
    const mm_hal_t* hal = mm_hal_mock();
    assert(hal != nullptr);

    /* malloc / free */
    void* ptr = nullptr;
    assert(hal->malloc(&ptr, 1024) == MM_OK);
    assert(ptr != nullptr);
    assert(hal->free(ptr) == MM_OK);

    /* malloc zero size => error */
    void* bad = nullptr;
    assert(hal->malloc(&bad, 0) == MM_ERR_INVALID_ARGUMENT);

    /* memcpy round-trip */
    void* gpu = nullptr;
    assert(hal->malloc(&gpu, 256) == MM_OK);

    unsigned char src[256];
    for (int i = 0; i < 256; i++)
        src[i] = (unsigned char)i;

    assert(hal->memcpy_h2d(gpu, src, 256, nullptr) == MM_OK);

    unsigned char dst[256] = {};
    assert(hal->memcpy_d2h(dst, gpu, 256, nullptr) == MM_OK);
    assert(memcmp(src, dst, 256) == 0);

    /* memset */
    assert(hal->memset(gpu, 0xAB, 256, nullptr) == MM_OK);
    unsigned char check[256] = {};
    assert(hal->memcpy_d2h(check, gpu, 256, nullptr) == MM_OK);
    for (int i = 0; i < 256; i++)
        assert(check[i] == 0xAB);

    assert(hal->free(gpu) == MM_OK);

    /* stream create / sync / destroy */
    mm_stream_t stream = nullptr;
    assert(hal->stream_create(&stream) == MM_OK);
    assert(stream != nullptr);
    assert(hal->stream_sync(stream) == MM_OK);
    assert(hal->stream_destroy(stream) == MM_OK);

    /* device queries */
    size_t free_bytes = 0, total_bytes = 0;
    assert(hal->get_free_mem(&free_bytes, &total_bytes) == MM_OK);
    assert(total_bytes > 0);

    assert(hal->set_device(0) == MM_OK);

    printf("test_hal_mock: PASSED\n");
    return 0;
}
