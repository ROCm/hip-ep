/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Natively-compiled KV cache tracking + MM singleton (Phase 4a). These live
// outside the bitcode-compiled memory_manager.cpp so the EP DLL's linker can
// resolve them. The MemoryManager class methods (register_kv_buffer etc.) are
// in bitcode; these wrappers bridge the gap via the process-level singleton.
//
// The singleton and KV tracking state are intentionally duplicated here (not
// shared with memory_manager.cpp's own kv_entries_) because bitcode and native
// code occupy separate link domains. The bitcode MM tracks KV buffers for
// in-graph accounting; these native wrappers track the same buffers for the
// EP DLL's cross-module access. Both are set/cleared in lockstep via the
// singleton lifecycle (set at init, cleared at cleanup).

#include <cstddef>
#include <cstdio>

// Process-level MemoryManager singleton (void* to avoid header dependency).
static void *g_mm_instance = nullptr;

// Simple KV buffer registry (native side). Mirrors MemoryManager::kv_entries_
// for cross-module access. Capacity matches MemoryManager::kMaxKvBuffers.
static constexpr int kMaxKvEntries = 256;
struct KvEntry {
  void *ptr = nullptr;
  size_t size = 0;
};
static KvEntry g_kv_entries[kMaxKvEntries] = {};
static int g_kv_count = 0;
static size_t g_kv_bytes = 0;

extern "C" void hipdnn_ep_mm_set_instance(void *mm) { g_mm_instance = mm; }

extern "C" void *hipdnn_ep_mm_get_instance() { return g_mm_instance; }

extern "C" void hipdnn_ep_mm_register_kv_buffer(void * /*mm*/, void *ptr,
                                                size_t size) {
  if (!ptr || size == 0)
    return;
  for (int i = 0; i < g_kv_count; ++i) {
    if (g_kv_entries[i].ptr == ptr)
      return;
  }
  if (g_kv_count >= kMaxKvEntries) {
    fprintf(stderr,
            "hipdnn_ep_mm_register_kv_buffer: capacity exceeded (%d)\n",
            kMaxKvEntries);
    return;
  }
  g_kv_entries[g_kv_count++] = {ptr, size};
  g_kv_bytes += size;
}

extern "C" void hipdnn_ep_mm_unregister_kv_buffer(void * /*mm*/, void *ptr) {
  if (!ptr)
    return;
  for (int i = 0; i < g_kv_count; ++i) {
    if (g_kv_entries[i].ptr == ptr) {
      g_kv_bytes -= g_kv_entries[i].size;
      g_kv_entries[i] = g_kv_entries[--g_kv_count];
      return;
    }
  }
}

extern "C" size_t hipdnn_ep_mm_kv_bytes_used() { return g_kv_bytes; }

extern "C" int hipdnn_ep_mm_kv_buffer_count() { return g_kv_count; }
