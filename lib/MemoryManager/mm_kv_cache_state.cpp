/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_kv_cache_state.h"
#include "mm_hal.h"

#include <cstdio>
#include <unordered_map>

struct mm_kv_cache_state {
  /* Map from host pointer to persistent GPU entry.
     Typical size: 64 entries for a 32-layer model (32 keys + 32 values). */
  std::unordered_map<const void *, mm_kv_persistent_entry_t> entries;
};

extern "C" {

mm_kv_cache_state_t *mm_kv_cache_state_create(void) {
  return new (std::nothrow) mm_kv_cache_state_t();
}

void mm_kv_cache_state_destroy(mm_kv_cache_state_t *state) {
  if (!state)
    return;

  /* Free all persistent GPU buffers */
  const mm_hal_t *hal = mm_hal_get();
  for (auto &[host_ptr, entry] : state->entries) {
    if (entry.gpu_ptr && hal) {
      hal->raw_free(0, entry.gpu_ptr);
    }
  }
  state->entries.clear();

  delete state;
}

mm_kv_persistent_entry_t *mm_kv_cache_state_lookup(mm_kv_cache_state_t *state,
                                                   const void *host_ptr) {
  if (!state || !host_ptr)
    return nullptr;

  auto it = state->entries.find(host_ptr);
  if (it == state->entries.end())
    return nullptr;

  return &it->second;
}

int mm_kv_cache_state_register(mm_kv_cache_state_t *state, void *host_ptr,
                               void *gpu_ptr, size_t size_bytes) {
  if (!state || !host_ptr || !gpu_ptr)
    return MM_ERROR_INVALID_ARG;

  /* Don't register duplicates */
  if (state->entries.count(host_ptr))
    return MM_OK;

  mm_kv_persistent_entry_t entry = {};
  entry.host_ptr = host_ptr;
  entry.gpu_ptr = gpu_ptr;
  entry.size_bytes = size_bytes;
  entry.inference_count = 0;

  state->entries[host_ptr] = entry;
  return MM_OK;
}

void mm_kv_cache_state_end_inference(mm_kv_cache_state_t *state) {
  if (!state)
    return;
  /* Nothing needed per-inference for now.
     inference_count is incremented per-entry in finalize_output. */
}

size_t mm_kv_cache_state_entry_count(const mm_kv_cache_state_t *state) {
  return state ? state->entries.size() : 0;
}

} /* extern "C" */
