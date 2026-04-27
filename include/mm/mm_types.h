/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * mm_types.h — Core type definitions for the Unified Memory Manager (UMM).
 *
 * This header defines the vocabulary types shared across all UMM components:
 * handles, memory classification, lifetime hints, allocation metadata, and
 * metrics. It has no dependencies beyond C99 stdint/stddef.
 */

#ifndef MM_TYPES_H
#define MM_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * mm_handle_t — Opaque allocation handle
 *
 * Every allocation returns a unique 64-bit handle rather than a raw pointer.
 * This indirection enables future transparent defragmentation (remap virtual
 * address without invalidating the handle) and tiered storage migration (same
 * handle, different physical location).
 *
 * Handles are monotonically increasing and never reused, which makes double-
 * free detection trivial (remove returns false if handle not found).
 * --------------------------------------------------------------------------- */
typedef uint64_t mm_handle_t;

/** Sentinel value indicating an invalid or failed allocation. */
#define MM_HANDLE_INVALID ((mm_handle_t)0)

/* ---------------------------------------------------------------------------
 * mm_stream_t — Opaque GPU stream handle
 *
 * Wraps the backend-specific stream type (e.g., hipStream_t on ROCm).
 * Pass NULL for synchronous operations or when the stream is not relevant.
 * --------------------------------------------------------------------------- */
typedef void* mm_stream_t;

/* ---------------------------------------------------------------------------
 * mm_device_t — Device identifier
 *
 * Non-negative values identify GPU ordinals (0, 1, ...).
 * MM_DEVICE_HOST (-1) identifies host (CPU) memory.
 * --------------------------------------------------------------------------- */
typedef int32_t mm_device_t;
#define MM_DEVICE_GPU0 0
#define MM_DEVICE_HOST (-1)

/* ---------------------------------------------------------------------------
 * mm_class_t — Memory classification
 *
 * Classifies the purpose of an allocation so the memory manager can route it
 * to a specialized pool in future stages. For the foundation layer, the class
 * is stored as metadata but does not affect allocation behavior.
 *
 *   GENERIC    — Unclassified allocation (default).
 *   WEIGHT     — Model weights. Static, read-only at inference time.
 *   ACTIVATION — Intermediate tensors. Short-lived, arena-allocated.
 *   KV_CACHE   — Attention key-value cache. Request-scoped, paged blocks.
 *   SCRATCH    — Kernel workspace. Per-stream, reusable ring buffer.
 * --------------------------------------------------------------------------- */
typedef enum {
    MM_CLASS_GENERIC    = 0,
    MM_CLASS_WEIGHT     = 1,
    MM_CLASS_ACTIVATION = 2,
    MM_CLASS_KV_CACHE   = 3,
    MM_CLASS_SCRATCH    = 4
} mm_class_t;

/* ---------------------------------------------------------------------------
 * mm_lifetime_t — Allocation lifetime hint
 *
 * Advises the memory manager about expected allocation duration, enabling
 * future optimizations (e.g., arena bulk-reset for STEP allocations).
 *
 *   STATIC    — Lives for the entire model session.
 *   REQUEST   — Lives for one inference request (seconds to minutes).
 *   STEP      — Lives for one decode step (freed after each generated token).
 *   TRANSIENT — Lives for a single kernel or fused operator group.
 * --------------------------------------------------------------------------- */
typedef enum {
    MM_LIFETIME_STATIC    = 0,
    MM_LIFETIME_REQUEST   = 1,
    MM_LIFETIME_STEP      = 2,
    MM_LIFETIME_TRANSIENT = 3
} mm_lifetime_t;

/* ---------------------------------------------------------------------------
 * mm_alloc_hints_t — Allocation hints passed to mm_alloc()
 *
 * Carries classification and alignment requirements. All fields are advisory
 * in the foundation layer; specialized pools will use them for routing.
 *
 * Fields:
 *   mem_class  — Memory classification (default: MM_CLASS_GENERIC).
 *   lifetime   — Expected lifetime (default: MM_LIFETIME_TRANSIENT).
 *   alignment  — Required byte alignment. 0 means use the config default
 *                (typically 256 bytes, matching GPU cache line size).
 * --------------------------------------------------------------------------- */
typedef struct {
    mm_class_t    mem_class;
    mm_lifetime_t lifetime;
    size_t        alignment;
} mm_alloc_hints_t;

/* ---------------------------------------------------------------------------
 * mm_alloc_info_t — Allocation metadata returned by mm_query()
 *
 * Describes a live allocation. All fields are set at allocation time and
 * remain constant for the lifetime of the handle.
 *
 * Fields:
 *   handle   — The unique handle for this allocation.
 *   ptr      — Raw device pointer (from HAL malloc).
 *   size     — Allocated size in bytes (may be larger than requested due to
 *              alignment rounding).
 *   mem_class — The memory class specified at allocation time.
 *   lifetime  — The lifetime hint specified at allocation time.
 *   device    — The device on which memory was allocated.
 * --------------------------------------------------------------------------- */
typedef struct {
    mm_handle_t   handle;
    void*         ptr;
    size_t        size;
    mm_class_t    mem_class;
    mm_lifetime_t lifetime;
    mm_device_t   device;
} mm_alloc_info_t;

/* ---------------------------------------------------------------------------
 * mm_metrics_snapshot_t — Point-in-time metrics snapshot
 *
 * Returned by mm_metrics_snapshot(). All values are read from atomic counters,
 * providing a consistent (though not transactional) view of allocator state.
 *
 * Fields:
 *   total_allocated_bytes — Sum of sizes of all currently live allocations.
 *   peak_allocated_bytes  — High-water mark of total_allocated_bytes.
 *   alloc_count           — Total number of mm_alloc() calls since init/reset.
 *   free_count            — Total number of mm_free() calls since init/reset.
 *   active_count          — Number of currently live allocations
 *                           (alloc_count - free_count).
 * --------------------------------------------------------------------------- */
typedef struct {
    size_t   total_allocated_bytes;
    size_t   peak_allocated_bytes;
    uint64_t alloc_count;
    uint64_t free_count;
    uint64_t active_count;
} mm_metrics_snapshot_t;

#ifdef __cplusplus
}
#endif

#endif /* MM_TYPES_H */
