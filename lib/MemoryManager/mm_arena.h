/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MM_ARENA_H
#define MM_ARENA_H

#include "mm_hal.h"
#include "mm_types.h"

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace mm {

/*
 * Size-class arena bump allocator with BFC fallback.
 *
 * 8 size classes covering exponential ranges:
 *   Class 0: [0, 1 KB)
 *   Class 1: [1 KB, 4 KB)
 *   Class 2: [4 KB, 16 KB)
 *   Class 3: [16 KB, 64 KB)
 *   Class 4: [64 KB, 256 KB)
 *   Class 5: [256 KB, 1 MB)
 *   Class 6: [1 MB, 4 MB)
 *   Class 7: [4 MB+) → BFC fallback
 *
 * Lock-free bump allocation via atomic fetch-add.
 * Step-scoped reset: arena.reset() = O(1).
 */

static constexpr size_t kArenaClassCount = 8;
static constexpr size_t kBfcThreshold = 4 * 1024 * 1024; /* 4 MB */
static constexpr size_t kDefaultAlignment = 256;

/* Class boundaries (upper limits, exclusive) */
static constexpr size_t kClassBounds[kArenaClassCount] = {
    1024,            /* 1 KB */
    4 * 1024,        /* 4 KB */
    16 * 1024,       /* 16 KB */
    64 * 1024,       /* 64 KB */
    256 * 1024,      /* 256 KB */
    1024 * 1024,     /* 1 MB */
    4 * 1024 * 1024, /* 4 MB */
    SIZE_MAX,        /* BFC */
};

/*
 * BFC (Best-Fit with Coalescing) allocator for large allocations.
 * Simple free-list based: best-fit allocation, coalesce adjacent on free.
 */
class BfcAllocator {
public:
  BfcAllocator() = default;
  ~BfcAllocator();

  int init(const mm_hal_t *hal, int device_id, size_t capacity);
  void *alloc(size_t size, size_t alignment = kDefaultAlignment);
  void free(void *ptr);
  void shutdown();

  size_t used_bytes() const { return used_bytes_; }
  size_t total_bytes() const { return capacity_; }

private:
  struct Block {
    size_t offset;
    size_t size;
    bool in_use;
  };

  const mm_hal_t *hal_ = nullptr;
  int device_id_ = 0;
  void *base_ = nullptr;
  size_t capacity_ = 0;
  size_t used_bytes_ = 0;
  std::mutex mutex_;
  std::vector<Block> blocks_;
};

class ArenaAllocator {
public:
  ArenaAllocator() = default;
  ~ArenaAllocator();

  /*
   * Initialize with HAL backend and per-class capacities.
   * total_budget is split across size classes proportionally.
   */
  int init(const mm_hal_t *hal, int device_id, size_t total_budget,
           uint32_t num_classes = kArenaClassCount);

  /*
   * Allocate from the appropriate size class.
   * Lock-free bump for classes 0-6, BFC for class 7.
   */
  void *alloc(size_t size, size_t alignment = kDefaultAlignment);

  /* Free a previous allocation. For arenas, this is a no-op until reset. */
  void free(void *ptr, size_t size);

  /* Reset all arenas to base. O(1). Used at step boundaries. */
  void reset();

  /* Shutdown and release all memory. */
  void shutdown();

  /* Determine which size class a given size falls into. */
  static uint32_t size_class(size_t size);

  size_t used_bytes() const;
  size_t total_bytes() const;

private:
  struct Arena {
    void *base = nullptr;
    size_t capacity = 0;
    std::atomic<size_t> offset{0};
  };

  const mm_hal_t *hal_ = nullptr;
  int device_id_ = 0;
  uint32_t num_classes_ = 0;
  Arena arenas_[kArenaClassCount];
  BfcAllocator bfc_;
  bool initialized_ = false;
};

} // namespace mm

#endif /* MM_ARENA_H */
