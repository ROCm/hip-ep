#include "mm_arena.h"

#include <algorithm>
#include <cstring>

namespace mm {

/* ---- BfcAllocator ---- */

BfcAllocator::~BfcAllocator() { shutdown(); }

int BfcAllocator::init(const mm_hal_t *hal, int device_id, size_t capacity) {
  if (!hal || capacity == 0)
    return MM_ERROR_INVALID_ARG;

  hal_ = hal;
  device_id_ = device_id;
  capacity_ = capacity;

  base_ = hal_->raw_alloc(device_id_, capacity_, kDefaultAlignment);
  if (!base_)
    return MM_ERROR_OUT_OF_MEMORY;

  /* Start with one large free block covering the entire slab */
  blocks_.push_back({0, capacity_, false});
  used_bytes_ = 0;
  return MM_OK;
}

void *BfcAllocator::alloc(size_t size, size_t alignment) {
  if (size == 0)
    return nullptr;

  /* Round up size to alignment */
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  std::lock_guard<std::mutex> lock(mutex_);

  /* Best-fit search */
  int best_idx = -1;
  size_t best_size = SIZE_MAX;
  for (int i = 0; i < (int)blocks_.size(); ++i) {
    if (!blocks_[i].in_use && blocks_[i].size >= aligned_size) {
      if (blocks_[i].size < best_size) {
        best_size = blocks_[i].size;
        best_idx = i;
      }
    }
  }

  if (best_idx < 0)
    return nullptr; /* Out of memory */

  Block &blk = blocks_[best_idx];
  void *ptr = static_cast<char *>(base_) + blk.offset;

  /* Split if remainder is large enough */
  if (blk.size > aligned_size + kDefaultAlignment) {
    Block remainder{blk.offset + aligned_size, blk.size - aligned_size, false};
    blk.size = aligned_size;
    blocks_.insert(blocks_.begin() + best_idx + 1, remainder);
  }

  blk.in_use = true;
  used_bytes_ += blk.size;
  return ptr;
}

void BfcAllocator::free(void *ptr) {
  if (!ptr || !base_)
    return;

  size_t offset = static_cast<char *>(ptr) - static_cast<char *>(base_);

  std::lock_guard<std::mutex> lock(mutex_);

  for (int i = 0; i < (int)blocks_.size(); ++i) {
    if (blocks_[i].offset == offset && blocks_[i].in_use) {
      blocks_[i].in_use = false;
      used_bytes_ -= blocks_[i].size;

      /* Coalesce with next block */
      if (i + 1 < (int)blocks_.size() && !blocks_[i + 1].in_use) {
        blocks_[i].size += blocks_[i + 1].size;
        blocks_.erase(blocks_.begin() + i + 1);
      }
      /* Coalesce with previous block */
      if (i > 0 && !blocks_[i - 1].in_use) {
        blocks_[i - 1].size += blocks_[i].size;
        blocks_.erase(blocks_.begin() + i);
      }
      return;
    }
  }
}

void BfcAllocator::shutdown() {
  if (base_ && hal_) {
    hal_->raw_free(device_id_, base_);
    base_ = nullptr;
  }
  blocks_.clear();
  capacity_ = 0;
  used_bytes_ = 0;
}

/* ---- ArenaAllocator ---- */

ArenaAllocator::~ArenaAllocator() { shutdown(); }

uint32_t ArenaAllocator::size_class(size_t size) {
  for (uint32_t c = 0; c < kArenaClassCount - 1; ++c) {
    if (size < kClassBounds[c])
      return c;
  }
  return kArenaClassCount - 1; /* BFC fallback */
}

int ArenaAllocator::init(const mm_hal_t *hal, int device_id,
                         size_t total_budget, uint32_t num_classes) {
  if (!hal || total_budget == 0)
    return MM_ERROR_INVALID_ARG;

  hal_ = hal;
  device_id_ = device_id;
  num_classes_ = std::min(num_classes, (uint32_t)kArenaClassCount);

  /*
   * Budget split: 70% to BFC (large allocs), 30% split across arenas.
   * This matches typical LLM workloads where workspace allocations
   * dominate activation memory.
   */
  size_t bfc_budget = (total_budget * 7) / 10;
  size_t arena_budget = total_budget - bfc_budget;
  size_t per_arena = arena_budget / num_classes_;

  /* Initialize per-class arenas */
  for (uint32_t c = 0; c < num_classes_; ++c) {
    /* Larger classes get proportionally more memory */
    size_t capacity = per_arena * (c + 1) / num_classes_;
    if (capacity < 4096)
      capacity = 4096;

    arenas_[c].base = hal_->raw_alloc(device_id_, capacity, kDefaultAlignment);
    if (!arenas_[c].base) {
      shutdown();
      return MM_ERROR_OUT_OF_MEMORY;
    }
    arenas_[c].capacity = capacity;
    arenas_[c].offset.store(0, std::memory_order_relaxed);
  }

  /* Initialize BFC for large allocations */
  if (bfc_budget > 0) {
    int err = bfc_.init(hal_, device_id_, bfc_budget);
    if (err != MM_OK) {
      shutdown();
      return err;
    }
  }

  initialized_ = true;
  return MM_OK;
}

void *ArenaAllocator::alloc(size_t size, size_t alignment) {
  if (!initialized_ || size == 0)
    return nullptr;

  uint32_t cls = size_class(size);

  /* BFC fallback for large allocations or class 7 */
  if (cls >= num_classes_ || size >= kBfcThreshold)
    return bfc_.alloc(size, alignment);

  /* Atomic bump allocation (lock-free) */
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
  Arena &arena = arenas_[cls];

  size_t old_offset =
      arena.offset.fetch_add(aligned_size, std::memory_order_relaxed);
  if (old_offset + aligned_size > arena.capacity) {
    /* Arena exhausted — fall back to BFC */
    arena.offset.fetch_sub(aligned_size, std::memory_order_relaxed);
    return bfc_.alloc(size, alignment);
  }

  return static_cast<char *>(arena.base) + old_offset;
}

void ArenaAllocator::free(void *ptr, size_t size) {
  if (!ptr)
    return;

  /* Check if this pointer belongs to BFC (not in any arena range) */
  for (uint32_t c = 0; c < num_classes_; ++c) {
    char *base = static_cast<char *>(arenas_[c].base);
    if (ptr >= base && ptr < base + arenas_[c].capacity)
      return; /* Arena allocs are freed in bulk via reset() */
  }

  /* Must be a BFC allocation */
  bfc_.free(ptr);
}

void ArenaAllocator::reset() {
  for (uint32_t c = 0; c < num_classes_; ++c) {
    arenas_[c].offset.store(0, std::memory_order_relaxed);
  }
  /* BFC allocations are not reset — they have explicit lifetimes */
}

void ArenaAllocator::shutdown() {
  if (!hal_)
    return;

  for (uint32_t c = 0; c < kArenaClassCount; ++c) {
    if (arenas_[c].base) {
      hal_->raw_free(device_id_, arenas_[c].base);
      arenas_[c].base = nullptr;
      arenas_[c].capacity = 0;
      arenas_[c].offset.store(0, std::memory_order_relaxed);
    }
  }

  bfc_.shutdown();
  initialized_ = false;
}

size_t ArenaAllocator::used_bytes() const {
  size_t total = 0;
  for (uint32_t c = 0; c < num_classes_; ++c) {
    total += arenas_[c].offset.load(std::memory_order_relaxed);
  }
  total += bfc_.used_bytes();
  return total;
}

size_t ArenaAllocator::total_bytes() const {
  size_t total = 0;
  for (uint32_t c = 0; c < num_classes_; ++c) {
    total += arenas_[c].capacity;
  }
  total += bfc_.total_bytes();
  return total;
}

} // namespace mm
