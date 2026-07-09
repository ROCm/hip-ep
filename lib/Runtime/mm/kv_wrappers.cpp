/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// C-linkage wrappers for KV cache buffer tracking (Phase 4a). These are
// compiled natively (not as bitcode) and linked into the EP DLL so the
// allocator module (morphizen/ort-bridge) can call them at link time.

#include "memory_manager.h"

extern "C" void hipdnn_ep_mm_register_kv_buffer(void *mm, void *ptr,
                                                size_t size) {
  if (mm)
    static_cast<MemoryManager *>(mm)->register_kv_buffer(ptr, size);
}

extern "C" void hipdnn_ep_mm_unregister_kv_buffer(void *mm, void *ptr) {
  if (mm)
    static_cast<MemoryManager *>(mm)->unregister_kv_buffer(ptr);
}

extern "C" void *hipdnn_ep_mm_get_instance() {
  return MemoryManager::get_instance();
}
