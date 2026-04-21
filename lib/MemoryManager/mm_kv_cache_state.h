/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MM_KV_CACHE_STATE_H
#define MM_KV_CACHE_STATE_H

#include "mm_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Persistent GPU buffer entry for KV cache.
 *
 * When OGA uses past_present_share_buffer=true, the same host pointer
 * appears as both input (past_key) and output (present_key). We detect
 * this and maintain a single persistent GPU buffer, eliminating H2D/D2H
 * copies after the first inference call.
 */
typedef struct {
  void *host_ptr;      /* OGA's stable OrtValue data pointer */
  void *gpu_ptr;       /* Persistent GPU buffer */
  size_t size_bytes;   /* Allocated buffer size */
  int inference_count; /* 0=first call (prefill), 1+=decode */
} mm_kv_persistent_entry_t;

/*
 * Opaque handle to the KV cache state.
 * Created once per session, persists across inference_compute() calls.
 */
typedef struct mm_kv_cache_state mm_kv_cache_state_t;

/* Create a new KV cache state. Returns NULL on failure. */
mm_kv_cache_state_t *mm_kv_cache_state_create(void);

/* Destroy and free all persistent GPU buffers. */
void mm_kv_cache_state_destroy(mm_kv_cache_state_t *state);

/*
 * Look up a persistent entry by host pointer.
 * Returns NULL if not registered.
 */
mm_kv_persistent_entry_t *mm_kv_cache_state_lookup(mm_kv_cache_state_t *state,
                                                   const void *host_ptr);

/*
 * Register a new persistent GPU buffer for the given host pointer.
 * Called after the first H2D copy to keep the GPU buffer alive.
 * Returns 0 on success.
 */
int mm_kv_cache_state_register(mm_kv_cache_state_t *state, void *host_ptr,
                               void *gpu_ptr, size_t size_bytes);

/*
 * Signal that one full inference iteration has completed.
 * Called at the end of each inference_compute().
 */
void mm_kv_cache_state_end_inference(mm_kv_cache_state_t *state);

/*
 * Get the number of registered persistent entries.
 */
size_t mm_kv_cache_state_entry_count(const mm_kv_cache_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* MM_KV_CACHE_STATE_H */
