#include "mm_pool.h"
#include "mm_internal.h"

#include <cstring>
#include <vector>

namespace mm {

class StaticPoolManager {
public:
  StaticPoolManager() = default;
  ~StaticPoolManager() { destroy(); }

  int create(const mm_hal_t *hal, const mm_static_plan_t *plan) {
    if (!hal || !plan || plan->num_entries == 0)
      return MM_ERROR_INVALID_ARG;

    hal_ = hal;
    device_id_ = (plan->device == MM_DEVICE_CPU) ? 0 : (int)plan->device;
    total_size_ = plan->total_size;

    /* Single HAL allocation */
    base_ptr_ =
        hal_->raw_alloc(device_id_, total_size_, 256 /* default alignment */);
    if (!base_ptr_)
      return MM_ERROR_OUT_OF_MEMORY;

    /* Build O(1) offset lookup table */
    entries_.resize(plan->num_entries);
    for (uint32_t i = 0; i < plan->num_entries; ++i) {
      entries_[plan->entries[i].tensor_id] = {
          plan->entries[i].offset,
          plan->entries[i].size,
          plan->entries[i].alignment,
      };
    }

    return MM_OK;
  }

  void *get_ptr(uint32_t tensor_id) {
    if (tensor_id >= entries_.size() || !base_ptr_)
      return nullptr;
    return static_cast<char *>(base_ptr_) + entries_[tensor_id].offset;
  }

  void destroy() {
    if (base_ptr_ && hal_) {
      hal_->raw_free(device_id_, base_ptr_);
      base_ptr_ = nullptr;
    }
    entries_.clear();
  }

  size_t total_size() const { return total_size_; }

private:
  struct Entry {
    size_t offset;
    size_t size;
    size_t alignment;
  };

  const mm_hal_t *hal_ = nullptr;
  int device_id_ = 0;
  void *base_ptr_ = nullptr;
  size_t total_size_ = 0;
  std::vector<Entry> entries_;
};

} // namespace mm

extern "C" {

mm_pool_t mm_create_pool(const mm_static_plan_t *plan) {
  auto *s = mm::mm_get_state();
  if (!s->initialized.load(std::memory_order_acquire))
    return MM_INVALID_POOL;
  if (!plan)
    return MM_INVALID_POOL;

  auto *pool = new mm::StaticPoolManager();
  int err = pool->create(s->hal, plan);
  if (err != MM_OK) {
    delete pool;
    return MM_INVALID_POOL;
  }

  /* Register the pool and return its index + 1 as handle */
  std::lock_guard<std::mutex> lock(s->pools_mutex);
  s->pools.push_back(pool);
  return static_cast<mm_pool_t>(s->pools.size());
}

void *mm_pool_get_ptr(mm_pool_t pool, uint32_t tensor_id) {
  auto *s = mm::mm_get_state();
  if (pool == MM_INVALID_POOL || pool == 0)
    return nullptr;

  std::lock_guard<std::mutex> lock(s->pools_mutex);
  size_t idx = pool - 1;
  if (idx >= s->pools.size() || !s->pools[idx])
    return nullptr;

  return s->pools[idx]->get_ptr(tensor_id);
}

void mm_destroy_pool(mm_pool_t pool) {
  auto *s = mm::mm_get_state();
  if (pool == MM_INVALID_POOL || pool == 0)
    return;

  std::lock_guard<std::mutex> lock(s->pools_mutex);
  size_t idx = pool - 1;
  if (idx >= s->pools.size() || !s->pools[idx])
    return;

  delete s->pools[idx];
  s->pools[idx] = nullptr;
}

} /* extern "C" */
