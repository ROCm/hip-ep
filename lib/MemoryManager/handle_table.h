/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef MM_HANDLE_TABLE_H
#define MM_HANDLE_TABLE_H

#include "mm/mm_types.h"
#include <mutex>
#include <unordered_map>

class HandleTable {
public:
  /** Insert a new allocation record. Returns the assigned handle. */
  mm_handle_t insert(void *ptr, size_t size, mm_class_t mem_class,
                     mm_lifetime_t lifetime, mm_device_t device);

  /** Look up by handle. Returns true and fills *info if found. */
  bool lookup(mm_handle_t handle, mm_alloc_info_t *info) const;

  /** Remove by handle. Returns true if found, false on double-free. */
  bool remove(mm_handle_t handle);

  /** Number of active allocations. */
  size_t size() const;

  /** Iterate all active allocations. Holds the lock for the duration. */
  template <typename Fn> void for_each(Fn &&fn) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &kv : table_)
      fn(kv.second);
  }

  /** Remove all entries (used by mm_shutdown). */
  void clear();

private:
  mutable std::mutex mutex_;
  std::unordered_map<mm_handle_t, mm_alloc_info_t> table_;
  uint64_t next_handle_ = 1;
};

#endif /* MM_HANDLE_TABLE_H */
