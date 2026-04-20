#ifndef MM_CORE_H
#define MM_CORE_H

#include "mm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocate memory with classification hints.
 * Routes to the appropriate sub-allocator based on mem_class:
 *   ACTIVATION → arena bump allocator
 *   SCRATCH    → arena (small) or BFC (large)
 *   WEIGHT     → error (use mm_create_pool instead)
 *   KV_CACHE   → error (use mm_kv_alloc_block instead, Phase 2)
 *
 * Returns MM_INVALID_HANDLE on failure.
 */
mm_handle_t mm_alloc(size_t size, mm_alloc_hints_t hints, mm_stream_t stream);

/*
 * Release an allocation. The handle becomes invalid after this call.
 */
void mm_free(mm_handle_t handle, mm_stream_t stream);

/*
 * Get a device-accessible pointer for the allocation.
 * Returns NULL if handle is invalid.
 */
void *mm_get_ptr(mm_handle_t handle, mm_device_t device);

/*
 * Prefetch an allocation to a specific tier (async).
 */
void mm_prefetch(mm_handle_t handle, mm_tier_t target_tier);

/*
 * Query allocation metadata.
 */
mm_alloc_info_t mm_query(mm_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* MM_CORE_H */
