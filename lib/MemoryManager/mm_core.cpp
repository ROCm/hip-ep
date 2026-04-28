/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "handle_table.h"
#include "mm/mm_api.h"
#include "mm/mm_hal.h"

#include <atomic>
#include <cstdio>
#include <cstring>

static const char *class_name(mm_class_t c) {
  switch (c) {
  case MM_CLASS_GENERIC:
    return "GENERIC";
  case MM_CLASS_WEIGHT:
    return "WEIGHT";
  case MM_CLASS_ACTIVATION:
    return "ACTIVATION";
  case MM_CLASS_KV_CACHE:
    return "KV_CACHE";
  case MM_CLASS_SCRATCH:
    return "SCRATCH";
  default:
    return "UNKNOWN";
  }
}

static const char *lifetime_name(mm_lifetime_t l) {
  switch (l) {
  case MM_LIFETIME_STATIC:
    return "STATIC";
  case MM_LIFETIME_REQUEST:
    return "REQUEST";
  case MM_LIFETIME_STEP:
    return "STEP";
  case MM_LIFETIME_TRANSIENT:
    return "TRANSIENT";
  default:
    return "UNKNOWN";
  }
}

/* Global state — file-scoped, not exported. */
static struct {
  const mm_hal_t *hal;
  HandleTable handles;
  mm_config_t config;
  bool initialized;

  std::atomic<size_t> total_allocated_bytes{0};
  std::atomic<size_t> peak_allocated_bytes{0};
  std::atomic<uint64_t> alloc_count{0};
  std::atomic<uint64_t> free_count{0};
} g_mm;

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static void update_peak() {
  size_t current = g_mm.total_allocated_bytes.load(std::memory_order_relaxed);
  size_t peak = g_mm.peak_allocated_bytes.load(std::memory_order_relaxed);
  while (current > peak) {
    if (g_mm.peak_allocated_bytes.compare_exchange_weak(
            peak, current, std::memory_order_relaxed))
      break;
  }
}

/* ============================= Lifecycle ================================= */

mm_status_t mm_init(const mm_config_t *config) {
  if (g_mm.initialized)
    return MM_ERR_ALREADY_INIT;

  if (config)
    g_mm.config = *config;
  else
    g_mm.config = mm_config_default();

#ifdef MM_USE_MOCK_HAL
  g_mm.hal = mm_hal_mock();
#else
  g_mm.hal = mm_hal_rocm();
#endif

  mm_status_t st = g_mm.hal->set_device(g_mm.config.device_id);
  if (st != MM_OK)
    return st;

  g_mm.total_allocated_bytes.store(0, std::memory_order_relaxed);
  g_mm.peak_allocated_bytes.store(0, std::memory_order_relaxed);
  g_mm.alloc_count.store(0, std::memory_order_relaxed);
  g_mm.free_count.store(0, std::memory_order_relaxed);

  g_mm.initialized = true;

  if (g_mm.config.enable_debug_log)
    fprintf(stderr, "[UMM] initialized (device=%d, alignment=%zu)\n",
            g_mm.config.device_id, g_mm.config.default_alignment);

  return MM_OK;
}

void mm_shutdown(void) {
  if (!g_mm.initialized)
    return;

  size_t leaked = g_mm.handles.size();
  if (leaked > 0) {
    if (g_mm.config.enable_debug_log)
      fprintf(stderr, "[UMM] shutdown: freeing %zu leaked allocation(s)\n",
              leaked);

    g_mm.handles.for_each([](const mm_alloc_info_t &info) {
      if (g_mm.config.enable_debug_log)
        fprintf(stderr,
                "[UMM]   leaked: handle=%llu ptr=%p size=%zu class=%s\n",
                (unsigned long long)info.handle, info.ptr, info.size,
                class_name(info.mem_class));
      g_mm.hal->free(info.ptr);
    });
  }

  g_mm.handles.clear();
  g_mm.total_allocated_bytes.store(0, std::memory_order_relaxed);
  g_mm.peak_allocated_bytes.store(0, std::memory_order_relaxed);
  g_mm.alloc_count.store(0, std::memory_order_relaxed);
  g_mm.free_count.store(0, std::memory_order_relaxed);
  g_mm.hal = nullptr;
  g_mm.initialized = false;
}

int mm_is_initialized(void) { return g_mm.initialized ? 1 : 0; }

/* ============================ Allocation ================================= */

mm_handle_t mm_alloc(size_t size, const mm_alloc_hints_t *hints,
                     mm_stream_t /*stream*/) {
  if (!g_mm.initialized)
    return MM_HANDLE_INVALID;
  if (size == 0)
    return MM_HANDLE_INVALID;

  mm_class_t mem_class = MM_CLASS_GENERIC;
  mm_lifetime_t lifetime = MM_LIFETIME_TRANSIENT;
  size_t alignment = g_mm.config.default_alignment;

  if (hints) {
    mem_class = hints->mem_class;
    lifetime = hints->lifetime;
    if (hints->alignment > 0)
      alignment = hints->alignment;
  }

  size_t aligned_size = align_up(size, alignment);

  void *ptr = nullptr;
  mm_status_t st = g_mm.hal->malloc(&ptr, aligned_size);
  if (st != MM_OK || !ptr)
    return MM_HANDLE_INVALID;

  mm_handle_t handle = g_mm.handles.insert(ptr, aligned_size, mem_class,
                                           lifetime, g_mm.config.device_id);

  g_mm.total_allocated_bytes.fetch_add(aligned_size, std::memory_order_relaxed);
  g_mm.alloc_count.fetch_add(1, std::memory_order_relaxed);
  update_peak();

  if (g_mm.config.enable_debug_log)
    fprintf(
        stderr, "[UMM] alloc: handle=%llu size=%zu (aligned=%zu) class=%s\n",
        (unsigned long long)handle, size, aligned_size, class_name(mem_class));

  return handle;
}

mm_status_t mm_free(mm_handle_t handle, mm_stream_t /*stream*/) {
  if (!g_mm.initialized)
    return MM_ERR_NOT_INITIALIZED;
  if (handle == MM_HANDLE_INVALID)
    return MM_ERR_INVALID_HANDLE;

  mm_alloc_info_t info;
  if (!g_mm.handles.lookup(handle, &info))
    return MM_ERR_INVALID_HANDLE;

  if (!g_mm.handles.remove(handle))
    return MM_ERR_DOUBLE_FREE;

  g_mm.hal->free(info.ptr);
  g_mm.total_allocated_bytes.fetch_sub(info.size, std::memory_order_relaxed);
  g_mm.free_count.fetch_add(1, std::memory_order_relaxed);

  if (g_mm.config.enable_debug_log)
    fprintf(stderr, "[UMM] free: handle=%llu size=%zu class=%s\n",
            (unsigned long long)handle, info.size, class_name(info.mem_class));

  return MM_OK;
}

void *mm_get_ptr(mm_handle_t handle) {
  if (!g_mm.initialized)
    return nullptr;

  mm_alloc_info_t info;
  if (!g_mm.handles.lookup(handle, &info))
    return nullptr;
  return info.ptr;
}

mm_status_t mm_query(mm_handle_t handle, mm_alloc_info_t *info) {
  if (!g_mm.initialized)
    return MM_ERR_NOT_INITIALIZED;
  if (!info)
    return MM_ERR_INVALID_ARGUMENT;

  if (!g_mm.handles.lookup(handle, info))
    return MM_ERR_INVALID_HANDLE;
  return MM_OK;
}

/* ============================ Diagnostics ================================ */

void mm_dump_state(FILE *output) {
  if (!output)
    return;

  mm_metrics_snapshot_t m = mm_metrics_snapshot();

  fprintf(output,
          "=== UMM State Dump ===\n"
          "  Initialized:  %s\n"
          "  Active allocs: %llu\n"
          "  Total bytes:   %zu\n"
          "  Peak bytes:    %zu\n"
          "  Alloc count:   %llu\n"
          "  Free count:    %llu\n"
          "  ---\n",
          g_mm.initialized ? "yes" : "no", (unsigned long long)m.active_count,
          m.total_allocated_bytes, m.peak_allocated_bytes,
          (unsigned long long)m.alloc_count, (unsigned long long)m.free_count);

  if (!g_mm.initialized)
    return;

  g_mm.handles.for_each([output](const mm_alloc_info_t &info) {
    fprintf(output,
            "  handle=%llu ptr=%p size=%zu class=%s lifetime=%s device=%d\n",
            (unsigned long long)info.handle, info.ptr, info.size,
            class_name(info.mem_class), lifetime_name(info.lifetime),
            info.device);
  });

  fprintf(output, "======================\n");
}

/* ============================== Metrics ================================== */

mm_metrics_snapshot_t mm_metrics_snapshot(void) {
  mm_metrics_snapshot_t snap;
  snap.total_allocated_bytes =
      g_mm.total_allocated_bytes.load(std::memory_order_relaxed);
  snap.peak_allocated_bytes =
      g_mm.peak_allocated_bytes.load(std::memory_order_relaxed);
  snap.alloc_count = g_mm.alloc_count.load(std::memory_order_relaxed);
  snap.free_count = g_mm.free_count.load(std::memory_order_relaxed);
  snap.active_count = snap.alloc_count - snap.free_count;
  return snap;
}

void mm_metrics_reset(void) {
  g_mm.alloc_count.store(0, std::memory_order_relaxed);
  g_mm.free_count.store(0, std::memory_order_relaxed);
}
