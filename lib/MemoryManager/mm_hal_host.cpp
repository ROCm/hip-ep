#include "mm_hal.h"
#include <cstdlib>
#include <cstring>

/* CPU/host HAL backend — for mock runtime and unit tests. */

static int host_get_device_count(void) { return 1; }

static int host_get_device_info(int device_id, mm_device_info_t *info) {
    if (device_id != 0 || !info)
        return MM_ERROR_INVALID_ARG;
    std::strncpy(info->name, "CPU-Host", MM_DEVICE_NAME_MAX);
    info->type = MM_DEVICE_CPU;
    info->total_memory = (size_t)16 * 1024 * 1024 * 1024; /* 16 GB nominal */
    info->free_memory  = (size_t)16 * 1024 * 1024 * 1024;
    return MM_OK;
}

static void *host_raw_alloc(int /*device_id*/, size_t size, size_t alignment) {
    if (size == 0)
        return nullptr;
#ifdef _WIN32
    return _aligned_malloc(size, alignment > 0 ? alignment : 64);
#else
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment > 0 ? alignment : 64, size) != 0)
        return nullptr;
    return ptr;
#endif
}

static void host_raw_free(int /*device_id*/, void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void *host_stream_alloc(int device_id, size_t size,
                               mm_stream_t /*stream*/) {
    return host_raw_alloc(device_id, size, 64);
}

static void host_stream_free(int /*device_id*/, void *ptr,
                             mm_stream_t /*stream*/) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int host_async_copy(void *dst, const void *src, size_t size,
                           mm_copy_kind_t /*kind*/, mm_stream_t /*stream*/,
                           mm_fence_t * /*fence*/) {
    if (!dst || !src)
        return MM_ERROR_INVALID_ARG;
    std::memcpy(dst, src, size);
    return MM_OK;
}

static int host_memset(void *ptr, int value, size_t size,
                       mm_stream_t /*stream*/) {
    if (!ptr)
        return MM_ERROR_INVALID_ARG;
    std::memset(ptr, value, size);
    return MM_OK;
}

static int host_stream_sync(mm_stream_t /*stream*/) { return MM_OK; }

static size_t host_get_total_memory(int /*device_id*/) {
    return (size_t)16 * 1024 * 1024 * 1024;
}

static size_t host_get_free_memory(int /*device_id*/) {
    return (size_t)16 * 1024 * 1024 * 1024;
}

static void *host_host_alloc(size_t size, size_t alignment) {
    return host_raw_alloc(0, size, alignment);
}

static void host_host_free(void *ptr) { host_raw_free(0, ptr); }

static const mm_hal_t g_host_hal = {
    host_get_device_count,
    host_get_device_info,
    host_raw_alloc,
    host_raw_free,
    host_stream_alloc,
    host_stream_free,
    host_async_copy,
    host_memset,
    host_stream_sync,
    host_get_total_memory,
    host_get_free_memory,
    host_host_alloc,
    host_host_free,
};

const mm_hal_t *mm_hal_host_get(void) { return &g_host_hal; }
