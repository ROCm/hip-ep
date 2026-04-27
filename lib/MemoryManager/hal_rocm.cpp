/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ROCm HAL backend — wraps HIP runtime API calls.
 * Only compiled when MM_USE_MOCK_HAL is NOT defined.
 */

#ifndef MM_USE_MOCK_HAL

#include "mm/mm_hal.h"
#include <hip/hip_runtime.h>

static mm_status_t rocm_malloc(void** ptr, size_t size) {
    if (!ptr || size == 0)
        return MM_ERR_INVALID_ARGUMENT;
    hipError_t err = hipMalloc(ptr, size);
    if (err == hipErrorOutOfMemory)
        return MM_ERR_OUT_OF_MEMORY;
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_free(void* ptr) {
    hipError_t err = hipFree(ptr);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_memcpy_h2d(void* dst, const void* src, size_t size,
                                   mm_stream_t stream) {
    if (!dst || !src)
        return MM_ERR_INVALID_ARGUMENT;
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, s);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_memcpy_d2h(void* dst, const void* src, size_t size,
                                   mm_stream_t stream) {
    if (!dst || !src)
        return MM_ERR_INVALID_ARGUMENT;
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, s);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_memset(void* ptr, int value, size_t size,
                               mm_stream_t stream) {
    if (!ptr)
        return MM_ERR_INVALID_ARGUMENT;
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemsetAsync(ptr, value, size, s);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_stream_create(mm_stream_t* stream) {
    if (!stream)
        return MM_ERR_INVALID_ARGUMENT;
    hipStream_t s;
    hipError_t err = hipStreamCreate(&s);
    if (err != hipSuccess)
        return MM_ERR_HAL_FAILURE;
    *stream = static_cast<mm_stream_t>(s);
    return MM_OK;
}

static mm_status_t rocm_stream_destroy(mm_stream_t stream) {
    hipError_t err = hipStreamDestroy(static_cast<hipStream_t>(stream));
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_stream_sync(mm_stream_t stream) {
    hipError_t err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_get_free_mem(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes)
        return MM_ERR_INVALID_ARGUMENT;
    hipError_t err = hipMemGetInfo(free_bytes, total_bytes);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

static mm_status_t rocm_set_device(mm_device_t device) {
    hipError_t err = hipSetDevice(device);
    return (err == hipSuccess) ? MM_OK : MM_ERR_HAL_FAILURE;
}

const mm_hal_t* mm_hal_rocm(void) {
    static const mm_hal_t vtable = {
        rocm_malloc,
        rocm_free,
        rocm_memcpy_h2d,
        rocm_memcpy_d2h,
        rocm_memset,
        rocm_stream_create,
        rocm_stream_destroy,
        rocm_stream_sync,
        rocm_get_free_mem,
        rocm_set_device
    };
    return &vtable;
}

#endif /* !MM_USE_MOCK_HAL */
