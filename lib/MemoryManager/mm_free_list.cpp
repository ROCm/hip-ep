#include "mm_free_list.h"

namespace mm {

FreeList::~FreeList() { shutdown(); }

void FreeList::init(uint32_t capacity) {
  capacity_ = capacity;
  /* Over-allocate: each block can be pushed at most once, but we need
     extra nodes if push/pop interleave heavily. 2x is safe. */
  uint32_t num_nodes = capacity * 2;
  nodes_ = new Node[num_nodes];
  for (uint32_t i = 0; i < num_nodes; ++i) {
    nodes_[i].block_index = UINT32_MAX;
    nodes_[i].next = UINT32_MAX;
  }
  head_.store(UINT32_MAX, std::memory_order_relaxed);
  free_nodes_head_.store(UINT32_MAX, std::memory_order_relaxed);
  next_node_.store(0, std::memory_order_relaxed);
  count_.store(0, std::memory_order_relaxed);
}

uint32_t FreeList::alloc_node() {
  /* Try the recycled node pool first */
  uint32_t idx = free_nodes_head_.load(std::memory_order_acquire);
  while (idx != UINT32_MAX) {
    uint32_t next = nodes_[idx].next;
    if (free_nodes_head_.compare_exchange_weak(
            idx, next, std::memory_order_acq_rel, std::memory_order_acquire))
      return idx;
  }
  /* Allocate from pre-allocated array */
  uint32_t new_idx = next_node_.fetch_add(1, std::memory_order_relaxed);
  if (new_idx >= capacity_ * 2)
    return UINT32_MAX; /* Exhausted */
  return new_idx;
}

void FreeList::free_node(uint32_t idx) {
  uint32_t old_head = free_nodes_head_.load(std::memory_order_acquire);
  do {
    nodes_[idx].next = old_head;
  } while (!free_nodes_head_.compare_exchange_weak(
      old_head, idx, std::memory_order_acq_rel, std::memory_order_acquire));
}

bool FreeList::push(uint32_t block_index) {
  uint32_t node_idx = alloc_node();
  if (node_idx == UINT32_MAX)
    return false;

  nodes_[node_idx].block_index = block_index;

  uint32_t old_head = head_.load(std::memory_order_acquire);
  do {
    nodes_[node_idx].next = old_head;
  } while (!head_.compare_exchange_weak(old_head, node_idx,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire));

  count_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

uint32_t FreeList::pop() {
  uint32_t idx = head_.load(std::memory_order_acquire);
  while (idx != UINT32_MAX) {
    uint32_t next = nodes_[idx].next;
    if (head_.compare_exchange_weak(idx, next, std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
      uint32_t block_index = nodes_[idx].block_index;
      free_node(idx);
      count_.fetch_sub(1, std::memory_order_relaxed);
      return block_index;
    }
  }
  return UINT32_MAX; /* Empty */
}

bool FreeList::empty() const {
  return head_.load(std::memory_order_acquire) == UINT32_MAX;
}

uint32_t FreeList::count() const {
  return count_.load(std::memory_order_relaxed);
}

void FreeList::shutdown() {
  delete[] nodes_;
  nodes_ = nullptr;
  capacity_ = 0;
  head_.store(UINT32_MAX, std::memory_order_relaxed);
  free_nodes_head_.store(UINT32_MAX, std::memory_order_relaxed);
  next_node_.store(0, std::memory_order_relaxed);
  count_.store(0, std::memory_order_relaxed);
}

} // namespace mm
