/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MM_HAL_H
#define MM_HAL_H

#include "mm_types.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MM_DEVICE_NAME_MAX 64

typedef struct {
  char name[MM_DEVICE_NAME_MAX];
  mm_device_t type;
  size_t total_memory;
  size_t free_memory;
} mm_device_info_t;

typedef struct {
  /* Device discovery */
  int (*get_device_count)(void);
  int (*get_device_info)(int device_id, mm_device_info_t *info);

  /* Raw allocation (synchronous) */
  void *(*raw_alloc)(int device_id, size_t size, size_t alignment);
  void (*raw_free)(int device_id, void *ptr);

  /* Stream-ordered allocation */
  void *(*stream_alloc)(int device_id, size_t size, mm_stream_t stream);
  void (*stream_free)(int device_id, void *ptr, mm_stream_t stream);

  /* Async data transfer */
  int (*async_copy)(void *dst, const void *src, size_t size,
                    mm_copy_kind_t kind, mm_stream_t stream, mm_fence_t *fence);

  /* Synchronous memset */
  int (*memset)(void *ptr, int value, size_t size, mm_stream_t stream);

  /* Stream synchronization */
  int (*stream_sync)(mm_stream_t stream);

  /* Device memory info */
  size_t (*get_total_memory)(int device_id);
  size_t (*get_free_memory)(int device_id);

  /* Host (pinned) memory for DRAM tier */
  void *(*host_alloc)(size_t size, size_t alignment);
  void (*host_free)(void *ptr);
} mm_hal_t;

/* Register a HAL backend. Must be called before mm_init(). */
int mm_hal_register(const mm_hal_t *hal);

/* Get the currently registered HAL (for internal use). */
const mm_hal_t *mm_hal_get(void);

/* Built-in HAL backends */
const mm_hal_t *mm_hal_host_get(void);

#ifdef MM_HAS_HIP
const mm_hal_t *mm_hal_hip_get(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MM_HAL_H */
