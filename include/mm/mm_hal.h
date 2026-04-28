/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef MM_HAL_H
#define MM_HAL_H

#include "mm_error.h"
#include "mm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * mm_hal_t — HAL function-pointer vtable
 *
 * Each function pointer wraps a backend-specific GPU operation. All functions
 * return MM_OK on success or an appropriate mm_status_t error code.
 *
 * Memory operations:
 *   malloc       — Allocate device memory. Sets *ptr to the allocated address.
 *   free         — Free device memory previously allocated by malloc.
 *   memcpy_h2d   — Copy from host to device (async if stream is non-NULL).
 *   memcpy_d2h   — Copy from device to host (async if stream is non-NULL).
 *   memset       — Fill device memory with a byte value.
 *
 * Stream operations:
 *   stream_create  — Create a new GPU stream.
 *   stream_destroy — Destroy a GPU stream.
 *   stream_sync    — Block until all operations on the stream complete.
 *
 * Device queries:
 *   get_free_mem — Query available and total device memory (in bytes).
 *   set_device   — Set the active GPU device for the calling thread.
 * ---------------------------------------------------------------------------
 */
typedef struct {
  mm_status_t (*malloc)(void **ptr, size_t size);
  mm_status_t (*free)(void *ptr);
  mm_status_t (*memcpy_h2d)(void *dst, const void *src, size_t size,
                            mm_stream_t stream);
  mm_status_t (*memcpy_d2h)(void *dst, const void *src, size_t size,
                            mm_stream_t stream);
  mm_status_t (*memset)(void *ptr, int value, size_t size, mm_stream_t stream);

  mm_status_t (*stream_create)(mm_stream_t *stream);
  mm_status_t (*stream_destroy)(mm_stream_t stream);
  mm_status_t (*stream_sync)(mm_stream_t stream);

  mm_status_t (*get_free_mem)(size_t *free_bytes, size_t *total_bytes);
  mm_status_t (*set_device)(mm_device_t device);
} mm_hal_t;

/**
 * Returns the ROCm HAL vtable.
 *
 * The returned pointer is to a static const singleton — do not free.
 * This backend calls hipMalloc, hipFree, hipMemcpyAsync, etc. and requires
 * a working ROCm/HIP installation.
 *
 * Only available when MM_USE_MOCK_HAL is NOT defined at build time.
 */
const mm_hal_t *mm_hal_rocm(void);

/**
 * Returns the mock HAL vtable.
 *
 * Uses malloc/free/memcpy for all operations. No GPU required.
 * Streams are dummy pointers (create allocates a byte, destroy frees it).
 * Useful for unit testing the memory manager on machines without a GPU.
 */
const mm_hal_t *mm_hal_mock(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_HAL_H */
