/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * KV cache block manager.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "mm/mm_config.h"
#include "mm/mm_types.h"

namespace mm {
namespace detail {

class Hal;
class HandleTable;

class KvManager {
public:
  KvManager() = default;

  Status init(Hal *hal, HandleTable *handles, const Config &config,
              std::size_t budget_bytes);
  void shutdown();

  handle_t alloc_block(const KvBlockDesc &desc,
                       std::size_t *reserved_bytes);
  Status free_block(handle_t handle, std::size_t *released_bytes);
  handle_t fork_block(handle_t source);
  bool build_block_table(const handle_t *handles, std::size_t count,
                         std::vector<void *> &out_ptrs) const;
  bool get_desc(handle_t handle, KvBlockDesc *out_desc) const;

  std::size_t total_bytes() const {
    return total_bytes_.load(std::memory_order_relaxed);
  }
  std::size_t peak_bytes() const {
    return peak_bytes_.load(std::memory_order_relaxed);
  }
  std::uint64_t handle_count() const {
    return handle_count_.load(std::memory_order_relaxed);
  }
  void reset_peak() {
    peak_bytes_.store(total_bytes_.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
  }

private:
  struct PhysicalEntry {
    std::size_t bytes = 0;
    KvBlockDesc desc{};
    std::uint32_t refcount = 0;
  };

  std::size_t compute_block_bytes(const KvBlockDesc &desc) const;
  std::size_t compute_bytes_per_token(const KvBlockDesc &desc) const;

  Hal *hal_ = nullptr;
  HandleTable *handles_ = nullptr;
  std::size_t budget_bytes_ = 0;

  mutable std::mutex mutex_;
  std::unordered_map<void *, PhysicalEntry> physical_;
  std::unordered_map<handle_t, void *> handle_to_ptr_;

  std::atomic<std::size_t> total_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
  std::atomic<std::uint64_t> handle_count_{0};
  std::size_t bytes_per_token_hint_ = 0;
  int device_id_ = 0;
};

} // namespace detail
} // namespace mm
