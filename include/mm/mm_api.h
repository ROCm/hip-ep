/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * mm_api.h — Public API for the Unified Memory Manager (UMM).
 *
 * Lifecycle:
 *   1. Call mm_init() once at startup with a config (or NULL for defaults).
 *   2. Use mm_alloc() / mm_free() / mm_get_ptr() / mm_query() during inference.
 *   3. Call mm_shutdown() at teardown to release all resources.
 *
 * Thread safety:
 *   All functions are thread-safe. The handle table uses a mutex internally.
 *   Metrics counters are atomic (lock-free reads via mm_metrics_snapshot()).
 *
 * Error handling:
 *   Functions that can fail return mm_status_t. Functions that return handles
 *   use MM_HANDLE_INVALID to signal failure. Check mm_status_string() for
 *   human-readable error descriptions.
 */

#ifndef MM_API_H
#define MM_API_H

#include "mm_types.h"
#include "mm_error.h"
#include "mm_config.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================= Lifecycle ================================= */

/**
 * Initialize the memory manager.
 *
 * Selects the HAL backend (ROCm or mock, depending on build configuration),
 * sets the active GPU device, and prepares internal data structures.
 *
 * Must be called before any other mm_* function. Call mm_shutdown() to
 * release resources. Calling mm_init() while already initialized returns
 * MM_ERR_ALREADY_INIT.
 *
 * @param config  Configuration parameters. Pass NULL to use defaults
 *                (device 0, 256-byte alignment, no debug logging).
 * @return MM_OK on success, MM_ERR_ALREADY_INIT if already initialized,
 *         or MM_ERR_HAL_FAILURE if the device could not be set.
 */
mm_status_t mm_init(const mm_config_t* config);

/**
 * Shut down the memory manager and release all resources.
 *
 * Any allocations still live at shutdown time are freed with a warning
 * printed to stderr (if debug logging is enabled). After this call, all
 * handles are invalid and mm_is_initialized() returns 0.
 *
 * Safe to call even if mm_init() was never called (no-op in that case).
 */
void mm_shutdown(void);

/**
 * Check whether the memory manager is initialized.
 *
 * @return 1 if mm_init() has been called and mm_shutdown() has not,
 *         0 otherwise.
 */
int mm_is_initialized(void);

/* ============================ Allocation ================================= */

/**
 * Allocate device memory and return a handle.
 *
 * The returned handle is an opaque 64-bit identifier. Use mm_get_ptr() to
 * obtain the raw device pointer when passing to GPU kernels.
 *
 * The actual allocation size may be larger than requested due to alignment
 * rounding. The allocated size is reported by mm_query().
 *
 * @param size   Bytes to allocate. Must be > 0.
 * @param hints  Classification and alignment metadata. Pass NULL for defaults
 *               (MM_CLASS_GENERIC, MM_LIFETIME_TRANSIENT, config alignment).
 * @param stream GPU stream for async operations (reserved for future arena
 *               allocators). Pass NULL for now.
 * @return A valid handle on success, MM_HANDLE_INVALID on failure (OOM,
 *         not initialized, or invalid arguments).
 */
mm_handle_t mm_alloc(size_t size, const mm_alloc_hints_t* hints,
                     mm_stream_t stream);

/**
 * Free a previously allocated handle.
 *
 * The underlying device memory is released back to the HAL. The handle
 * becomes invalid after this call.
 *
 * @param handle A valid handle returned by mm_alloc().
 * @param stream GPU stream for async operations (reserved). Pass NULL.
 * @return MM_OK on success, MM_ERR_INVALID_HANDLE if the handle was never
 *         allocated, MM_ERR_DOUBLE_FREE if already freed, or
 *         MM_ERR_NOT_INITIALIZED if the manager is not initialized.
 */
mm_status_t mm_free(mm_handle_t handle, mm_stream_t stream);

/**
 * Get the raw device pointer for a handle.
 *
 * The returned pointer is valid until mm_free() is called on the handle.
 *
 * @param handle A valid handle returned by mm_alloc().
 * @return The device pointer, or NULL if the handle is invalid.
 */
void* mm_get_ptr(mm_handle_t handle);

/**
 * Query metadata for an active allocation.
 *
 * @param handle A valid handle returned by mm_alloc().
 * @param info   Output parameter filled with allocation metadata.
 * @return MM_OK on success, MM_ERR_INVALID_HANDLE if the handle is not
 *         active, or MM_ERR_NOT_INITIALIZED.
 */
mm_status_t mm_query(mm_handle_t handle, mm_alloc_info_t* info);

/* ============================ Diagnostics ================================ */

/**
 * Dump all active allocations to a file stream (human-readable).
 *
 * Prints a summary header (total count, total bytes, peak bytes) followed
 * by one line per active allocation (handle, pointer, size, class, lifetime).
 *
 * @param output File stream to write to (e.g., stderr). Must not be NULL.
 */
void mm_dump_state(FILE* output);

/* ============================== Metrics ================================== */

/**
 * Take a point-in-time snapshot of allocator metrics.
 *
 * Reads atomic counters without locking. The snapshot is consistent per-field
 * but not transactionally consistent across fields (a concurrent alloc/free
 * may be partially reflected).
 *
 * @return A metrics snapshot struct with current gauge and counter values.
 */
mm_metrics_snapshot_t mm_metrics_snapshot(void);

/**
 * Reset the alloc/free counters to zero.
 *
 * The peak_allocated_bytes gauge is NOT reset (it tracks the all-time high).
 * Useful for per-request or per-batch metric collection.
 */
void mm_metrics_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_API_H */
