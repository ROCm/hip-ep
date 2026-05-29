/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_handle_table.h"

#include <limits>

namespace mm {
namespace detail {

namespace {
bool will_overflow(handle_t next) {
  return next == std::numeric_limits<handle_t>::max();
}
} // namespace

handle_t HandleTable::insert(void *ptr, std::size_t size, MemoryClass mem_class,
                             Lifetime lifetime, int device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (will_overflow(next_handle_))
    next_handle_ = 1;
  const handle_t handle = next_handle_++;
  AllocInfo info;
  info.handle = handle;
  info.ptr = ptr;
  info.size = size;
  info.mem_class = mem_class;
  info.lifetime = lifetime;
  info.device = device;
  entries_.emplace(handle, info);
  return handle;
}

bool HandleTable::remove(handle_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.erase(handle) == 1;
}

bool HandleTable::lookup(handle_t handle, AllocInfo *out_info) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(handle);
  if (it == entries_.end())
    return false;
  if (out_info)
    *out_info = it->second;
  return true;
}

std::size_t HandleTable::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

void HandleTable::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
}

} // namespace detail
} // namespace mm
