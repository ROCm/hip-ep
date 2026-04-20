#ifndef MM_POOL_H
#define MM_POOL_H

#include "mm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Create a static memory pool from a compiler-generated plan.
 * Performs a single HAL allocation of plan->total_size bytes and builds
 * an O(1) offset lookup table.
 * Returns MM_INVALID_POOL on failure.
 */
mm_pool_t mm_create_pool(const mm_static_plan_t *plan);

/*
 * Get the device pointer for a tensor within the pool.
 * O(1) array index lookup — no allocator call.
 * Returns NULL if pool or tensor_id is invalid.
 */
void *mm_pool_get_ptr(mm_pool_t pool, uint32_t tensor_id);

/*
 * Destroy a pool and release its underlying memory.
 */
void mm_destroy_pool(mm_pool_t pool);

#ifdef __cplusplus
}
#endif

#endif /* MM_POOL_H */
