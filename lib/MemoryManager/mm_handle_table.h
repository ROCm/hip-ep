/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MM_HANDLE_TABLE_H
#define MM_HANDLE_TABLE_H

#include "mm_types.h"

#include <atomic>
#include <mutex>
#include <vector>

namespace mm {

struct HandleEntry {
  void *ptr;
  size_t size;
  mm_memory_class_t mem_class;
  mm_tier_t tier;
  uint32_t ref_count;
  bool active;
};

class HandleTable {
public:
  HandleTable();

  /* Insert a new entry, returns a monotonically increasing handle. */
  mm_handle_t insert(void *ptr, size_t size, mm_memory_class_t mem_class,
                     mm_tier_t tier = MM_TIER_HBM);

  /* Lookup by handle. Returns copy. active=false if invalid/removed. */
  HandleEntry lookup(mm_handle_t handle);

  /* Remove an entry. Returns copy of entry before removal.
     active=false if handle was already invalid. */
  HandleEntry remove(mm_handle_t handle);

  /* Number of active entries. */
  size_t active_count() const;

  /* Clear all entries. */
  void clear();

private:
  mutable std::mutex mutex_;
  std::vector<HandleEntry> entries_;
  std::atomic<uint64_t> next_id_{1}; /* 0 is MM_INVALID_HANDLE */
};

} // namespace mm

#endif /* MM_HANDLE_TABLE_H */
