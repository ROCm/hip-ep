#include "mm_handle_table.h"

namespace mm {

HandleTable::HandleTable() = default;

mm_handle_t HandleTable::insert(void *ptr, size_t size,
                                mm_memory_class_t mem_class, mm_tier_t tier) {
  uint64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);

  HandleEntry entry{};
  entry.ptr = ptr;
  entry.size = size;
  entry.mem_class = mem_class;
  entry.tier = tier;
  entry.ref_count = 1;
  entry.active = true;

  std::lock_guard<std::mutex> lock(mutex_);
  /* Handle ID maps to index = id - 1 (since IDs start at 1) */
  if (id - 1 >= entries_.size())
    entries_.resize(id, HandleEntry{});
  entries_[id - 1] = entry;
  return static_cast<mm_handle_t>(id);
}

HandleEntry *HandleTable::lookup(mm_handle_t handle) {
  if (handle == MM_INVALID_HANDLE)
    return nullptr;
  uint64_t idx = handle - 1;
  std::lock_guard<std::mutex> lock(mutex_);
  if (idx >= entries_.size() || !entries_[idx].active)
    return nullptr;
  return &entries_[idx];
}

HandleEntry *HandleTable::remove(mm_handle_t handle) {
  if (handle == MM_INVALID_HANDLE)
    return nullptr;
  uint64_t idx = handle - 1;
  std::lock_guard<std::mutex> lock(mutex_);
  if (idx >= entries_.size() || !entries_[idx].active)
    return nullptr;
  entries_[idx].active = false;
  return &entries_[idx];
}

size_t HandleTable::active_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto &e : entries_) {
    if (e.active)
      ++count;
  }
  return count;
}

void HandleTable::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  next_id_.store(1, std::memory_order_relaxed);
}

} // namespace mm
