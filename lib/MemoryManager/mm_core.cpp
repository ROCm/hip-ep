/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_core.h"
#include "mm_arena.h"
#include "mm_internal.h"

extern "C" {

mm_handle_t mm_alloc(size_t size, mm_alloc_hints_t hints,
                     mm_stream_t /*stream*/) {
  auto *s = mm::mm_get_state();
  if (!s->initialized.load(std::memory_order_acquire))
    return MM_INVALID_HANDLE;

  void *ptr = nullptr;

  switch (hints.mem_class) {
  case MM_CLASS_ACTIVATION:
  case MM_CLASS_SCRATCH:
    if (!s->arena)
      return MM_INVALID_HANDLE;
    ptr = s->arena->alloc(size, hints.alignment > 0 ? hints.alignment : 256);
    break;

  case MM_CLASS_WEIGHT:
    /* Weights must use mm_create_pool() */
    return MM_INVALID_HANDLE;

  case MM_CLASS_KV_CACHE:
    /* KV cache must use mm_kv_alloc_block() */
    return MM_INVALID_HANDLE;

  default:
    return MM_INVALID_HANDLE;
  }

  if (!ptr)
    return MM_INVALID_HANDLE;

  return s->handles.insert(ptr, size, hints.mem_class);
}

void mm_free(mm_handle_t handle, mm_stream_t /*stream*/) {
  auto *s = mm::mm_get_state();
  if (!s->initialized.load(std::memory_order_acquire))
    return;

  mm::HandleEntry entry = s->handles.remove(handle);
  if (!entry.active)
    return;

  switch (entry.mem_class) {
  case MM_CLASS_ACTIVATION:
  case MM_CLASS_SCRATCH:
    if (s->arena)
      s->arena->free(entry.ptr, entry.size);
    break;
  default:
    break;
  }
}

void *mm_get_ptr(mm_handle_t handle, mm_device_t /*device*/) {
  auto *s = mm::mm_get_state();
  if (!s->initialized.load(std::memory_order_acquire))
    return nullptr;

  mm::HandleEntry entry = s->handles.lookup(handle);
  if (!entry.active)
    return nullptr;

  return entry.ptr;
}

void mm_prefetch(mm_handle_t /*handle*/, mm_tier_t /*target_tier*/) {
  /* Phase 3: tier migration */
}

mm_alloc_info_t mm_query(mm_handle_t handle) {
  mm_alloc_info_t info = {};

  auto *s = mm::mm_get_state();
  if (!s->initialized.load(std::memory_order_acquire))
    return info;

  mm::HandleEntry entry = s->handles.lookup(handle);
  if (!entry.active)
    return info;

  info.current_tier = entry.tier;
  info.mem_class = entry.mem_class;
  info.size = entry.size;
  info.ref_count = entry.ref_count;
  info.resident_device = MM_DEVICE_GPU_0;
  return info;
}

} /* extern "C" */
