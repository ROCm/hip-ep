/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm/mm_api.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "mm/mm_config.h"
#include "mm/mm_kv.h"
#include "mm_activation.h"
#include "mm_hal.h"
#include "mm_handle_table.h"
#include "mm_kv_manager.h"

namespace mm {
namespace {

using detail::ActivationArena;
using detail::Hal;
using detail::HandleTable;
using detail::KvManager;

struct ManagerState {
  Config config;
  HandleTable handles;
  ActivationArena activation;
  KvManager kv_manager;
  Hal *hal = nullptr;
  bool initialized = false;

  std::atomic<std::size_t> total_allocated{0};
  std::atomic<std::size_t> peak_allocated{0};
  std::atomic<std::uint64_t> alloc_count{0};
  std::atomic<std::uint64_t> free_count{0};
  std::atomic<std::uint64_t> kv_alloc_count{0};
  std::atomic<std::uint64_t> kv_free_count{0};
};

std::mutex g_mutex;
ManagerState g_state;

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || alignment == 1)
    return value;
  const std::size_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

void update_peak(std::size_t current) {
  std::size_t expected = g_state.peak_allocated.load(std::memory_order_relaxed);
  while (current > expected &&
         !g_state.peak_allocated.compare_exchange_weak(
             expected, current, std::memory_order_relaxed)) {
  }
}

} // namespace

Status init(const Config *config) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_state.initialized)
    return Status::ErrAlreadyInit;

  g_state.config = config ? *config : config_default();
  g_state.hal = detail::hal_rocm();
  if (!g_state.hal)
    return Status::ErrHalFailure;

  Status st = g_state.hal->set_device(g_state.config.device_id);
  if (st != Status::Ok)
    return st;

  st = g_state.kv_manager.init(g_state.hal, &g_state.handles, g_state.config);
  if (st != Status::Ok)
    return st;

  st = g_state.activation.init(g_state.hal, g_state.config);
  if (st != Status::Ok)
    return st;

  g_state.total_allocated.store(0, std::memory_order_relaxed);
  g_state.peak_allocated.store(0, std::memory_order_relaxed);
  g_state.alloc_count.store(0, std::memory_order_relaxed);
  g_state.free_count.store(0, std::memory_order_relaxed);
  g_state.kv_alloc_count.store(0, std::memory_order_relaxed);
  g_state.kv_free_count.store(0, std::memory_order_relaxed);
  g_state.initialized = true;
  return Status::Ok;
}

void shutdown() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return;

  g_state.kv_manager.shutdown();
  g_state.activation.shutdown(g_state.hal);

  g_state.handles.for_each([](const AllocInfo &info) {
    if (!g_state.hal || !info.ptr)
      return;
    if (info.mem_class == MemoryClass::KvCache ||
        info.mem_class == MemoryClass::Activation) {
      return;
    }
    (void)g_state.hal->free(info.ptr);
  });
  g_state.handles.clear();
  g_state.total_allocated.store(0, std::memory_order_relaxed);
  g_state.peak_allocated.store(0, std::memory_order_relaxed);
  g_state.alloc_count.store(0, std::memory_order_relaxed);
  g_state.free_count.store(0, std::memory_order_relaxed);
  g_state.kv_alloc_count.store(0, std::memory_order_relaxed);
  g_state.kv_free_count.store(0, std::memory_order_relaxed);
  g_state.hal = nullptr;
  g_state.initialized = false;
}

int is_initialized() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_state.initialized ? 1 : 0;
}

handle_t alloc(std::size_t size_bytes, const AllocHints *hints) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized || !g_state.hal || size_bytes == 0)
    return kInvalidHandle;

  AllocHints local_hints;
  if (hints)
    local_hints = *hints;

  const std::size_t alignment = local_hints.alignment
                                    ? local_hints.alignment
                                    : g_state.config.default_alignment;
  void *ptr = nullptr;
  std::size_t reserved_size = align_up(size_bytes, alignment);

  if (local_hints.mem_class == MemoryClass::Activation) {
    auto arena_alloc =
        g_state.activation.alloc(size_bytes, alignment, g_state.hal);
    if (arena_alloc.ptr) {
      ptr = arena_alloc.ptr;
      reserved_size = arena_alloc.reserved_size;
    }
  }

  if (!ptr) {
    Status st = g_state.hal->malloc(&ptr, reserved_size);
    if (st != Status::Ok || !ptr)
      return kInvalidHandle;
  }

  const handle_t handle =
      g_state.handles.insert(ptr, reserved_size, local_hints.mem_class,
                             local_hints.lifetime, g_state.config.device_id);

  const std::size_t total = g_state.total_allocated.fetch_add(
                                reserved_size, std::memory_order_relaxed) +
                            reserved_size;
  update_peak(total);
  g_state.alloc_count.fetch_add(1, std::memory_order_relaxed);
  return handle;
}

Status free(handle_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized || !g_state.hal)
    return Status::ErrNotInitialized;
  if (handle == kInvalidHandle)
    return Status::ErrInvalidHandle;

  AllocInfo info;
  if (!g_state.handles.lookup(handle, &info))
    return Status::ErrInvalidHandle;

  if (info.mem_class == MemoryClass::KvCache) {
    std::size_t released = 0;
    Status st = g_state.kv_manager.free_block(handle, &released);
    if (st == Status::Ok) {
      g_state.kv_free_count.fetch_add(1, std::memory_order_relaxed);
      if (released > 0)
        g_state.total_allocated.fetch_sub(released, std::memory_order_relaxed);
    }
    return st;
  }

  if (!g_state.handles.remove(handle))
    return Status::ErrDoubleFree;

  Status st = Status::Ok;
  if (info.mem_class == MemoryClass::Activation) {
    g_state.activation.release(info.ptr, g_state.hal);
  } else {
    st = g_state.hal->free(info.ptr);
    if (st != Status::Ok)
      return st;
  }

  g_state.total_allocated.fetch_sub(info.size, std::memory_order_relaxed);
  g_state.free_count.fetch_add(1, std::memory_order_relaxed);
  return Status::Ok;
}

void *get_ptr(handle_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return nullptr;
  AllocInfo info;
  if (!g_state.handles.lookup(handle, &info))
    return nullptr;
  return info.ptr;
}

Status query(handle_t handle, AllocInfo *info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return Status::ErrNotInitialized;
  if (!info)
    return Status::ErrInvalidArgument;
  if (!g_state.handles.lookup(handle, info))
    return Status::ErrInvalidHandle;
  return Status::Ok;
}

void dump_state(std::FILE *output) {
  if (!output)
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::fprintf(output,
               "=== Memory Manager State ===\n"
               "  Initialized: %s\n"
               "  Active allocs: %zu\n"
               "  Total bytes: %zu\n"
               "  Peak bytes: %zu\n"
               "  Alloc count: %llu\n"
               "  Free count: %llu\n",
               g_state.initialized ? "yes" : "no", g_state.handles.size(),
               g_state.total_allocated.load(std::memory_order_relaxed),
               g_state.peak_allocated.load(std::memory_order_relaxed),
               static_cast<unsigned long long>(
                   g_state.alloc_count.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(
                   g_state.free_count.load(std::memory_order_relaxed)));
  if (!g_state.initialized) {
    std::fprintf(output, "============================\n");
    return;
  }

  g_state.handles.for_each([output](const AllocInfo &info) {
    std::fprintf(
        output,
        "  handle=%llu ptr=%p size=%zu class=%u lifetime=%u device=%d\n",
        static_cast<unsigned long long>(info.handle), info.ptr, info.size,
        static_cast<unsigned>(info.mem_class),
        static_cast<unsigned>(info.lifetime), info.device);
  });
  std::fprintf(output, "============================\n");
}

int debug_enabled() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return (g_state.initialized && g_state.config.enable_debug_log) ? 1 : 0;
}

MetricsSnapshot metrics_snapshot() {
  MetricsSnapshot snap;
  snap.total_allocated_bytes =
      g_state.total_allocated.load(std::memory_order_relaxed);
  snap.peak_allocated_bytes =
      g_state.peak_allocated.load(std::memory_order_relaxed);
  snap.alloc_count = g_state.alloc_count.load(std::memory_order_relaxed);
  snap.free_count = g_state.free_count.load(std::memory_order_relaxed);
  snap.active_count = snap.alloc_count - snap.free_count;
  snap.activation_bytes = g_state.activation.total_bytes();
  snap.activation_peak_bytes = g_state.activation.peak_bytes();
  snap.kv_bytes = g_state.kv_manager.total_bytes();
  snap.kv_peak_bytes = g_state.kv_manager.peak_bytes();
  snap.kv_alloc_count = g_state.kv_alloc_count.load(std::memory_order_relaxed);
  snap.kv_free_count = g_state.kv_free_count.load(std::memory_order_relaxed);
  snap.kv_block_handle_count = g_state.kv_manager.handle_count();
  return snap;
}

void metrics_reset() {
  g_state.alloc_count.store(0, std::memory_order_relaxed);
  g_state.free_count.store(0, std::memory_order_relaxed);
  g_state.total_allocated.store(0, std::memory_order_relaxed);
  g_state.peak_allocated.store(0, std::memory_order_relaxed);
  g_state.kv_alloc_count.store(0, std::memory_order_relaxed);
  g_state.kv_free_count.store(0, std::memory_order_relaxed);
  g_state.activation.reset_peak();
  g_state.kv_manager.reset_peak();
}

kv_block_t kv_alloc_block(const KvBlockDesc &desc) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return kInvalidHandle;
  std::size_t reserved = 0;
  handle_t handle = g_state.kv_manager.alloc_block(desc, &reserved);
  if (handle == kInvalidHandle)
    return kInvalidHandle;
  g_state.kv_alloc_count.fetch_add(1, std::memory_order_relaxed);
  g_state.alloc_count.fetch_add(1, std::memory_order_relaxed);
  if (reserved > 0) {
    const std::size_t total =
        g_state.total_allocated.fetch_add(reserved, std::memory_order_relaxed) +
        reserved;
    update_peak(total);
  }
  return handle;
}

Status kv_free_block(kv_block_t block) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return Status::ErrNotInitialized;
  std::size_t released = 0;
  Status st = g_state.kv_manager.free_block(block, &released);
  if (st != Status::Ok)
    return st;
  g_state.kv_free_count.fetch_add(1, std::memory_order_relaxed);
  g_state.free_count.fetch_add(1, std::memory_order_relaxed);
  if (released > 0)
    g_state.total_allocated.fetch_sub(released, std::memory_order_relaxed);
  return Status::Ok;
}

kv_block_t kv_fork_block(kv_block_t source) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return kInvalidHandle;
  handle_t handle = g_state.kv_manager.fork_block(source);
  if (handle != kInvalidHandle) {
    g_state.kv_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_state.alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  return handle;
}

bool kv_get_block_table(const kv_block_t *blocks, std::size_t count,
                        void **out_ptrs) {
  if (!blocks || !out_ptrs)
    return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return false;
  std::vector<void *> table;
  bool ok = g_state.kv_manager.build_block_table(blocks, count, table);
  if (!ok || table.size() != count)
    return false;
  for (std::size_t i = 0; i < count; ++i)
    out_ptrs[i] = table[i];
  return true;
}

bool kv_get_block_desc(kv_block_t block, KvBlockDesc *out_desc) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_state.initialized)
    return false;
  return g_state.kv_manager.get_desc(block, out_desc);
}

} // namespace mm
