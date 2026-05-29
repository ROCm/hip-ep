/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Internal handle table for memory manager.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "mm/mm_types.h"

namespace mm {
namespace detail {

class HandleTable {
public:
  HandleTable() = default;

  handle_t insert(void *ptr, std::size_t size, MemoryClass mem_class,
                  Lifetime lifetime, int device);
  bool remove(handle_t handle);
  bool lookup(handle_t handle, AllocInfo *out_info) const;
  std::size_t size() const;
  void clear();

  template <typename Fn> void for_each(Fn &&fn) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &entry : entries_) {
      fn(entry.second);
    }
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<handle_t, AllocInfo> entries_;
  handle_t next_handle_ = 1;
};

} // namespace detail
} // namespace mm
