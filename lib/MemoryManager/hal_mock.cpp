/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mm/mm_hal.h"
#include <cstdlib>
#include <cstring>

static mm_status_t mock_malloc(void **ptr, size_t size) {
  if (!ptr || size == 0)
    return MM_ERR_INVALID_ARGUMENT;
  *ptr = std::malloc(size);
  return *ptr ? MM_OK : MM_ERR_OUT_OF_MEMORY;
}

static mm_status_t mock_free(void *ptr) {
  std::free(ptr);
  return MM_OK;
}

static mm_status_t mock_memcpy_h2d(void *dst, const void *src, size_t size,
                                   mm_stream_t /*stream*/) {
  if (!dst || !src)
    return MM_ERR_INVALID_ARGUMENT;
  std::memcpy(dst, src, size);
  return MM_OK;
}

static mm_status_t mock_memcpy_d2h(void *dst, const void *src, size_t size,
                                   mm_stream_t /*stream*/) {
  if (!dst || !src)
    return MM_ERR_INVALID_ARGUMENT;
  std::memcpy(dst, src, size);
  return MM_OK;
}

static mm_status_t mock_memset(void *ptr, int value, size_t size,
                               mm_stream_t /*stream*/) {
  if (!ptr)
    return MM_ERR_INVALID_ARGUMENT;
  std::memset(ptr, value, size);
  return MM_OK;
}

static mm_status_t mock_stream_create(mm_stream_t *stream) {
  if (!stream)
    return MM_ERR_INVALID_ARGUMENT;
  *stream = std::malloc(1);
  return *stream ? MM_OK : MM_ERR_OUT_OF_MEMORY;
}

static mm_status_t mock_stream_destroy(mm_stream_t stream) {
  std::free(stream);
  return MM_OK;
}

static mm_status_t mock_stream_sync(mm_stream_t /*stream*/) { return MM_OK; }

static mm_status_t mock_get_free_mem(size_t *free_bytes, size_t *total_bytes) {
  if (!free_bytes || !total_bytes)
    return MM_ERR_INVALID_ARGUMENT;
  *total_bytes = (size_t)16 * 1024 * 1024 * 1024; /* 16 GB simulated */
  *free_bytes = *total_bytes;
  return MM_OK;
}

static mm_status_t mock_set_device(mm_device_t /*device*/) { return MM_OK; }

const mm_hal_t *mm_hal_mock(void) {
  static const mm_hal_t vtable = {mock_malloc,         mock_free,
                                  mock_memcpy_h2d,     mock_memcpy_d2h,
                                  mock_memset,         mock_stream_create,
                                  mock_stream_destroy, mock_stream_sync,
                                  mock_get_free_mem,   mock_set_device};
  return &vtable;
}
