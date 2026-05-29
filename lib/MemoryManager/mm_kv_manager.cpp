/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mm_kv_manager.h"

#include <algorithm>

#include "mm_hal.h"
#include "mm_handle_table.h"

namespace mm {
namespace detail {

namespace {
void update_peak(std::atomic<std::size_t> &peak, std::size_t current) {
  std::size_t expected = peak.load(std::memory_order_relaxed);
  while (current > expected &&
         !peak.compare_exchange_weak(expected, current,
                                     std::memory_order_relaxed)) {
  }
}
} // namespace

Status KvManager::init(Hal *hal, HandleTable *handles, const Config &config,
                       std::size_t budget_bytes) {
  if (!hal || !handles)
    return Status::ErrInvalidArgument;
  hal_ = hal;
  handles_ = handles;
  budget_bytes_ = budget_bytes;
  physical_.clear();
  handle_to_ptr_.clear();
  total_bytes_.store(0, std::memory_order_relaxed);
  peak_bytes_.store(0, std::memory_order_relaxed);
  handle_count_.store(0, std::memory_order_relaxed);
  bytes_per_token_hint_ =
      config.kv_bytes_per_token_hint ? config.kv_bytes_per_token_hint : 0;
  device_id_ = config.device_id;
  return Status::Ok;
}

void KvManager::shutdown() {
  if (!hal_)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &entry : physical_) {
    (void)hal_->free(entry.first);
  }
  physical_.clear();
  handle_to_ptr_.clear();
  total_bytes_.store(0, std::memory_order_relaxed);
  peak_bytes_.store(0, std::memory_order_relaxed);
  handle_count_.store(0, std::memory_order_relaxed);
}

handle_t KvManager::alloc_block(const KvBlockDesc &desc,
                                std::size_t *reserved_bytes) {
  if (!hal_ || !handles_)
    return kInvalidHandle;

  const std::size_t block_bytes = compute_block_bytes(desc);
  if (block_bytes == 0)
    return kInvalidHandle;

  std::lock_guard<std::mutex> lock(mutex_);
  if (budget_bytes_ != 0 &&
      total_bytes_.load(std::memory_order_relaxed) + block_bytes >
          budget_bytes_) {
    return kInvalidHandle;
  }

  void *ptr = nullptr;
  Status st = hal_->malloc(&ptr, block_bytes);
  if (st != Status::Ok || !ptr)
    return kInvalidHandle;

  physical_[ptr] = PhysicalEntry{block_bytes, desc, 1};
  const handle_t handle = handles_->insert(
      ptr, block_bytes, MemoryClass::KvCache, Lifetime::Request, device_id_);
  handle_to_ptr_[handle] = ptr;
  total_bytes_.fetch_add(block_bytes, std::memory_order_relaxed);
  update_peak(peak_bytes_, total_bytes_.load(std::memory_order_relaxed));
  handle_count_.fetch_add(1, std::memory_order_relaxed);
  if (reserved_bytes)
    *reserved_bytes = block_bytes;
  return handle;
}

Status KvManager::free_block(handle_t handle, std::size_t *released_bytes) {
  if (!hal_ || !handles_)
    return Status::ErrNotInitialized;

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handle_to_ptr_.find(handle);
  if (it == handle_to_ptr_.end())
    return Status::ErrInvalidHandle;
  void *ptr = it->second;
  auto phys_it = physical_.find(ptr);
  if (phys_it == physical_.end())
    return Status::ErrInvalidHandle;

  handle_to_ptr_.erase(it);
  handle_count_.fetch_sub(1, std::memory_order_relaxed);
  if (!handles_->remove(handle))
    return Status::ErrInvalidHandle;

  PhysicalEntry &entry = phys_it->second;
  if (entry.refcount > 0)
    --entry.refcount;

  if (entry.refcount == 0) {
    total_bytes_.fetch_sub(entry.bytes, std::memory_order_relaxed);
    (void)hal_->free(ptr);
    physical_.erase(phys_it);
    if (released_bytes)
      *released_bytes = entry.bytes;
  } else {
    if (released_bytes)
      *released_bytes = 0;
  }

  return Status::Ok;
}

handle_t KvManager::fork_block(handle_t source) {
  if (!hal_ || !handles_)
    return kInvalidHandle;

  std::lock_guard<std::mutex> lock(mutex_);
  auto map_it = handle_to_ptr_.find(source);
  if (map_it == handle_to_ptr_.end())
    return kInvalidHandle;

  void *ptr = map_it->second;
  auto phys_it = physical_.find(ptr);
  if (phys_it == physical_.end())
    return kInvalidHandle;

  phys_it->second.refcount += 1;

  const handle_t handle =
      handles_->insert(ptr, phys_it->second.bytes, MemoryClass::KvCache,
                       Lifetime::Request, device_id_);
  handle_to_ptr_[handle] = ptr;
  handle_count_.fetch_add(1, std::memory_order_relaxed);
  return handle;
}

bool KvManager::build_block_table(const handle_t *handles, std::size_t count,
                                  std::vector<void *> &out_ptrs) const {
  out_ptrs.clear();
  if (!handles_)
    return false;

  std::lock_guard<std::mutex> lock(mutex_);
  out_ptrs.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto it = handle_to_ptr_.find(handles[i]);
    if (it == handle_to_ptr_.end())
      return false;
    out_ptrs.push_back(it->second);
  }
  return true;
}

bool KvManager::get_desc(handle_t handle, KvBlockDesc *out_desc) const {
  if (!out_desc)
    return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto map_it = handle_to_ptr_.find(handle);
  if (map_it == handle_to_ptr_.end())
    return false;
  auto phys_it = physical_.find(map_it->second);
  if (phys_it == physical_.end())
    return false;
  *out_desc = phys_it->second.desc;
  return true;
}

std::size_t KvManager::compute_block_bytes(const KvBlockDesc &desc) const {
  const std::size_t bytes_per_token = compute_bytes_per_token(desc);
  if (bytes_per_token == 0)
    return 0;
  return bytes_per_token *
         static_cast<std::size_t>(std::max<std::uint32_t>(
             static_cast<std::uint32_t>(1), desc.block_size_tokens));
}

std::size_t KvManager::compute_bytes_per_token(const KvBlockDesc &desc) const {
  if (desc.bytes_per_token != 0)
    return desc.bytes_per_token;

  if (bytes_per_token_hint_ != 0)
    return bytes_per_token_hint_;

  if (desc.num_kv_heads == 0 || desc.head_dim == 0 || desc.num_layers == 0)
    return 0;

  std::size_t element_bytes = 2;
  switch (desc.format) {
  case KvFormat::Fp16:
    element_bytes = 2;
    break;
  case KvFormat::Fp8E4M3:
    element_bytes = 1;
    break;
  case KvFormat::Int4:
    element_bytes = 1; // includes scale/zero overhead.
    break;
  case KvFormat::TurboQuant4:
    element_bytes = 1; // approximate.
    break;
  case KvFormat::TurboQuant3:
  case KvFormat::TurboQuant2:
    element_bytes = 1;
    break;
  default:
    element_bytes = 2;
    break;
  }

  const std::size_t heads = static_cast<std::size_t>(desc.num_kv_heads) *
                            static_cast<std::size_t>(desc.num_layers);
  const std::size_t per_token =
      heads * static_cast<std::size_t>(desc.head_dim) * element_bytes * 2;
  return per_token;
}

} // namespace detail
} // namespace mm
