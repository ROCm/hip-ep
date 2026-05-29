/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * Activation arena allocator for dynamic tensors.
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mm/mm_config.h"
#include "mm/mm_types.h"

namespace mm {
namespace detail {

class Hal;

class ActivationArena {
public:
  ActivationArena() = default;

  struct Allocation {
    void *ptr = nullptr;
    std::size_t reserved_size = 0;
  };

  Status init(Hal *hal, const Config &config);
  void shutdown(Hal *hal);

  Allocation alloc(std::size_t size, std::size_t alignment, Hal *hal);
  void release(void *ptr, Hal *hal);

  std::size_t total_bytes() const {
    return total_bytes_.load(std::memory_order_relaxed);
  }
  std::size_t peak_bytes() const {
    return peak_bytes_.load(std::memory_order_relaxed);
  }
  void reset_peak() {
    peak_bytes_.store(total_bytes_.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
  }
  bool is_initialized() const { return initialized_; }

private:
  struct Class {
    std::size_t upper_bound = 0;
    std::size_t alloc_size = 0;
    std::vector<void *> free_list;
    std::mutex mutex;
  };

  std::size_t find_class(std::size_t size) const;
  void update_peak(std::size_t current);

  std::array<Class, 8> classes_{};
  std::unordered_map<void *, std::size_t> allocation_sizes_;
  std::mutex allocations_mutex_;
  std::atomic<std::size_t> total_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
  bool initialized_ = false;
};

} // namespace detail
} // namespace mm
