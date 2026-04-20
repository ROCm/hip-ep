#ifndef MM_RUNTIME_BRIDGE_H
#define MM_RUNTIME_BRIDGE_H

/*
 * Bridge layer between the existing HipDNN EP runtime and the Unified
 * Memory Manager.
 *
 * This header provides drop-in replacements for the runtime's memory
 * operations. The integration strategy is:
 *
 * 1. The runtime calls mm_bridge_init() during hipdnn_ep_state_init_with_fs()
 *    to initialize the MM with the appropriate HAL backend.
 *
 * 2. pool_alloc/pool_release in hipdnn_ep_runtime_tensor.cpp are replaced
 *    with mm_bridge_tensor_alloc/mm_bridge_tensor_release.
 *
 * 3. hipdnn_ep_pool_init() calls mm_bridge_pool_init() to create a static
 *    pool via mm_create_pool().
 *
 * 4. hipdnn_ep_state_ensure_workspace() calls mm_bridge_workspace_alloc()
 *    to route workspace through mm_alloc(SCRATCH).
 *
 * 5. mm_bridge_shutdown() is called during hipdnn_ep_state_cleanup().
 *
 * All functions are C-linkage for compatibility with the generated LLVM IR.
 */

#include "mm_config.h"
#include "mm_core.h"
#include "mm_hal.h"
#include "mm_pool.h"
#include "mm_types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the MM bridge. Called once during state_init.
 * gpu_memory_limit: 0 for auto-detect.
 * hip_stream: the HIP stream cast to uint64_t (mm_stream_t).
 * Returns MM_OK on success.
 */
int mm_bridge_init(size_t gpu_memory_limit, mm_stream_t hip_stream);

/*
 * Shutdown the MM bridge. Called during state_cleanup.
 */
void mm_bridge_shutdown(void);

/*
 * Create a static pool from compiler-generated offsets.
 * Drop-in replacement for hipdnn_ep_pool_init().
 * Returns the mm_pool_t handle (stored in RuntimeState).
 */
mm_pool_t mm_bridge_pool_init(size_t pool_size,
                              const size_t *buffer_offsets,
                              size_t num_buffers);

/*
 * Get a buffer pointer from the static pool.
 * Drop-in replacement for hipdnn_ep_get_buffer_from_pool().
 */
void *mm_bridge_pool_get_buffer(mm_pool_t pool, size_t index);

/*
 * Get pool base pointer (for MemoryLowering.cpp compatibility).
 * Drop-in replacement for hipdnn_ep_get_pool_base().
 */
void *mm_bridge_pool_get_base(mm_pool_t pool);

/*
 * Destroy a static pool.
 */
void mm_bridge_pool_destroy(mm_pool_t pool);

/*
 * Allocate a GPU buffer for tensor I/O.
 * Drop-in replacement for pool_alloc() in hipdnn_ep_runtime_tensor.cpp.
 */
void *mm_bridge_tensor_alloc(size_t size_bytes);

/*
 * Release a GPU buffer for tensor I/O.
 * Drop-in replacement for pool_release() in hipdnn_ep_runtime_tensor.cpp.
 */
void mm_bridge_tensor_release(void *ptr, size_t size_bytes);

/*
 * Ensure workspace of at least needed_size bytes.
 * Drop-in replacement for hipdnn_ep_state_ensure_workspace().
 * Returns the workspace pointer, or NULL on failure.
 */
void *mm_bridge_workspace_ensure(size_t needed_size);

/*
 * Get current workspace pointer (for operator use).
 */
void *mm_bridge_workspace_get(void);

/*
 * Get current workspace size.
 */
size_t mm_bridge_workspace_size(void);

/*
 * Check if MM bridge is initialized.
 */
int mm_bridge_is_initialized(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_RUNTIME_BRIDGE_H */
