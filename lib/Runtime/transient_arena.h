/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_TRANSIENT_ARENA_H
#define HIPDNN_EP_RUNTIME_TRANSIENT_ARENA_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <optional>

namespace hipdnn_ep {

// First-fit offset allocator over one fixed-size region (the session's
// transient arena). It hands out kAlign-aligned offsets and coalesces freed
// blocks with their neighbours; the caller owns the backing memory and maps
// offsets to pointers. A request that does not fit is counted as a miss so the
// caller can fall back to a plain device allocation.
class TransientArenaAllocator {
public:
  static constexpr size_t kAlign = 4096;

  static size_t roundUp(size_t bytes) {
    if (bytes == 0)
      return kAlign;
    return (bytes + kAlign - 1) & ~(kAlign - 1);
  }

  void init(size_t size) {
    size_ = size & ~(kAlign - 1);
    free_.clear();
    live_.clear();
    used_ = hwm_ = hits_ = misses_ = miss_bytes_ = 0;
    if (size_ > 0)
      free_[0] = size_;
  }

  std::optional<size_t> allocate(size_t bytes) {
    if (bytes <= size_) {
      const size_t need = roundUp(bytes);
      for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->second < need)
          continue;
        const size_t offset = it->first;
        const size_t avail = it->second;
        free_.erase(it);
        if (avail > need)
          free_[offset + need] = avail - need;
        live_[offset] = need;
        used_ += need;
        hwm_ = std::max(hwm_, used_);
        ++hits_;
        return offset;
      }
    }
    ++misses_;
    miss_bytes_ += bytes;
    return std::nullopt;
  }

  // False when offset is not the start of a live block.
  bool release(size_t offset) {
    auto live = live_.find(offset);
    if (live == live_.end())
      return false;
    size_t len = live->second;
    live_.erase(live);
    used_ -= len;
    auto next = free_.lower_bound(offset);
    if (next != free_.end() && next->first == offset + len) {
      len += next->second;
      next = free_.erase(next);
    }
    if (next != free_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == offset) {
        prev->second += len;
        return true;
      }
    }
    free_[offset] = len;
    return true;
  }

  // Bytes held by the live block starting at offset, or 0.
  size_t blockBytes(size_t offset) const {
    auto live = live_.find(offset);
    return live == live_.end() ? 0 : live->second;
  }

  size_t size() const { return size_; }
  size_t used() const { return used_; }
  size_t highWaterMark() const { return hwm_; }
  size_t hits() const { return hits_; }
  size_t misses() const { return misses_; }
  size_t missBytes() const { return miss_bytes_; }
  size_t freeBlockCount() const { return free_.size(); }

private:
  size_t size_ = 0;
  std::map<size_t, size_t> free_; // offset -> bytes
  std::map<size_t, size_t> live_; // offset -> bytes
  size_t used_ = 0, hwm_ = 0, hits_ = 0, misses_ = 0, miss_bytes_ = 0;
};

} // namespace hipdnn_ep

#endif // HIPDNN_EP_RUNTIME_TRANSIENT_ARENA_H
