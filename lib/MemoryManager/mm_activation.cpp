/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_activation.h"

#include <algorithm>
#include <limits>

#include "mm_hal.h"

namespace mm {
namespace detail {

namespace {
std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || alignment == 1)
    return value;
  const std::size_t mask = alignment - 1;
  return (value + mask) & ~mask;
}
} // namespace

Status ActivationArena::init(Hal *hal, const Config &config,
                             std::size_t budget_bytes) {
  (void)hal;
  budget_bytes_ = budget_bytes;
  allocation_sizes_.clear();
  for (auto &clazz : classes_) {
    std::lock_guard<std::mutex> lock(clazz.mutex);
    clazz.free_list.clear();
  }

  for (std::size_t i = 0; i < classes_.size(); ++i) {
    classes_[i].upper_bound = config.activation_size_class_upper_bounds[i];
    if (i < classes_.size() - 1)
      classes_[i].alloc_size = config.activation_size_class_upper_bounds[i];
    else
      classes_[i].alloc_size = 0; // fallback class uses request size.
  }

  total_bytes_.store(0, std::memory_order_relaxed);
  peak_bytes_.store(0, std::memory_order_relaxed);
  initialized_ = true;
  return Status::Ok;
}

void ActivationArena::shutdown(Hal *hal) {
  if (!initialized_)
    return;

  for (std::size_t i = 0; i < classes_.size(); ++i) {
    auto &clazz = classes_[i];
    std::lock_guard<std::mutex> lock(clazz.mutex);
    const std::size_t alloc_size =
        (clazz.alloc_size == 0) ? 0 : clazz.alloc_size;
    for (void *ptr : clazz.free_list) {
      if (alloc_size != 0)
        total_bytes_.fetch_sub(alloc_size, std::memory_order_relaxed);
      (void)hal->free(ptr);
    }
    clazz.free_list.clear();
  }

  {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    for (auto &entry : allocation_sizes_) {
      total_bytes_.fetch_sub(entry.second, std::memory_order_relaxed);
      (void)hal->free(entry.first);
    }
    allocation_sizes_.clear();
  }

  total_bytes_.store(0, std::memory_order_relaxed);
  peak_bytes_.store(0, std::memory_order_relaxed);
  initialized_ = false;
}

std::size_t ActivationArena::find_class(std::size_t size) const {
  for (std::size_t i = 0; i < classes_.size(); ++i) {
    if (size <= classes_[i].upper_bound)
      return i;
  }
  return classes_.size() - 1;
}

void ActivationArena::update_peak(std::size_t current) {
  std::size_t expected = peak_bytes_.load(std::memory_order_relaxed);
  while (current > expected &&
         !peak_bytes_.compare_exchange_weak(expected, current,
                                            std::memory_order_relaxed)) {
  }
}

ActivationArena::Allocation
ActivationArena::alloc(std::size_t size, std::size_t alignment, Hal *hal) {
  Allocation result;
  if (!initialized_ || !hal)
    return result;

  const std::size_t class_index = find_class(size);
  Class &clazz = classes_[class_index];

  const bool fallback_class =
      (class_index == classes_.size() - 1) || clazz.alloc_size == 0 ||
      clazz.upper_bound == clazz.alloc_size &&
          clazz.upper_bound == std::numeric_limits<std::size_t>::max();

  std::size_t alloc_size = fallback_class ? size : clazz.alloc_size;
  alloc_size = align_up(alloc_size, alignment);

  void *ptr = nullptr;
  bool counted_budget = false;

  if (!fallback_class) {
    {
      std::lock_guard<std::mutex> lock(clazz.mutex);
      if (!clazz.free_list.empty()) {
        ptr = clazz.free_list.back();
        clazz.free_list.pop_back();
      }
    }
    if (!ptr) {
      if (budget_bytes_ != 0) {
        std::size_t total = total_bytes_.load(std::memory_order_relaxed);
        while (true) {
          if (total + alloc_size > budget_bytes_)
            return result;
          if (total_bytes_.compare_exchange_weak(total, total + alloc_size,
                                                 std::memory_order_relaxed)) {
            counted_budget = true;
            break;
          }
        }
      } else {
        total_bytes_.fetch_add(alloc_size, std::memory_order_relaxed);
        counted_budget = true;
      }

      Status st = hal->malloc(&ptr, alloc_size);
      if (st != Status::Ok || !ptr) {
        if (counted_budget)
          total_bytes_.fetch_sub(alloc_size, std::memory_order_relaxed);
        return result;
      }
      update_peak(total_bytes_.load(std::memory_order_relaxed));
    }
  } else {
    Status st = hal->malloc(&ptr, alloc_size);
    if (st != Status::Ok || !ptr)
      return result;
    total_bytes_.fetch_add(alloc_size, std::memory_order_relaxed);
    update_peak(total_bytes_.load(std::memory_order_relaxed));
  }

  {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    allocation_sizes_[ptr] = alloc_size;
  }

  result.ptr = ptr;
  result.reserved_size = alloc_size;
  return result;
}

void ActivationArena::release(void *ptr, Hal *hal) {
  if (!ptr || !hal)
    return;

  std::size_t alloc_size = 0;
  {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    auto it = allocation_sizes_.find(ptr);
    if (it == allocation_sizes_.end())
      return;
    alloc_size = it->second;
    allocation_sizes_.erase(it);
  }

  const std::size_t class_index = find_class(alloc_size);
  Class &clazz = classes_[class_index];
  const bool fallback_class =
      (class_index == classes_.size() - 1) || clazz.alloc_size == 0 ||
      clazz.upper_bound == clazz.alloc_size &&
          clazz.upper_bound == std::numeric_limits<std::size_t>::max();

  if (fallback_class) {
    total_bytes_.fetch_sub(alloc_size, std::memory_order_relaxed);
    (void)hal->free(ptr);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(clazz.mutex);
    clazz.free_list.push_back(ptr);
  }
}

} // namespace detail
} // namespace mm
