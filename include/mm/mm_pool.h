/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef MM_POOL_H
#define MM_POOL_H

#include "mm_error.h"
#include "mm_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque pool handle. */
typedef struct mm_pool_s *mm_pool_t;

/* ---------------------------------------------------------------------------
 * mm_static_plan_t — Compile-time memory plan
 *
 * Describes how a single GPU allocation is subdivided into sub-buffers.
 * Produced by the MLIR PoolAllocs pass and stored as module attributes
 * (hipdnn.pool_size, hipdnn.buffer_offsets, hipdnn.buffer_count).
 *
 * Fields:
 *   total_size   — Total bytes to allocate (single hipMalloc).
 *   offsets      — Array of byte offsets for each sub-buffer within the pool.
 *                  Caller retains ownership; the pool copies this array.
 *   num_entries  — Number of sub-buffers (length of offsets array).
 * ---------------------------------------------------------------------------
 */
typedef struct {
  size_t total_size;
  const size_t *offsets;
  uint32_t num_entries;
} mm_static_plan_t;

/**
 * Create a static pool from a compile-time memory plan.
 *
 * Performs a single mm_alloc() of plan->total_size bytes, then stores the
 * offset table for O(1) sub-buffer lookup. The memory manager must be
 * initialized (mm_init called) before calling this function.
 *
 * @param plan  The memory plan. Must not be NULL. plan->total_size must be > 0.
 * @return Pool handle on success, NULL on failure (OOM or invalid args).
 */
mm_pool_t mm_pool_create(const mm_static_plan_t *plan);

/**
 * Get the raw device pointer for a sub-buffer by index.
 *
 * Returns base + offsets[index]. O(1) operation.
 *
 * @param pool   Pool handle returned by mm_pool_create().
 * @param index  Sub-buffer index (0-based, must be < num_entries).
 * @return Device pointer, or NULL if pool is NULL or index is out of bounds.
 */
void *mm_pool_get_ptr(mm_pool_t pool, uint32_t index);

/**
 * Get the pool base pointer.
 *
 * This is the raw pointer from the single underlying allocation. Used by
 * hip.get_pool lowering in generated compute kernels.
 *
 * @param pool  Pool handle.
 * @return Base device pointer, or NULL if pool is NULL.
 */
void *mm_pool_get_base(mm_pool_t pool);

/**
 * Get the total pool size in bytes.
 *
 * @param pool  Pool handle.
 * @return Total size, or 0 if pool is NULL.
 */
size_t mm_pool_get_size(mm_pool_t pool);

/**
 * Get the number of sub-buffer entries in the pool.
 *
 * @param pool  Pool handle.
 * @return Number of entries, or 0 if pool is NULL.
 */
uint32_t mm_pool_get_num_entries(mm_pool_t pool);

/**
 * Destroy a static pool and free the underlying GPU memory.
 *
 * Calls mm_free() on the underlying allocation and frees the offset table.
 * Safe to call with NULL (no-op). After this call, the pool handle is invalid.
 *
 * @param pool  Pool handle to destroy, or NULL.
 */
void mm_pool_destroy(mm_pool_t pool);

#ifdef __cplusplus
}
#endif

#endif /* MM_POOL_H */
