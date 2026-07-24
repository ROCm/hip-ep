/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- block_pool.h - Physical KV Cache Block Pool (Phase 4b) ------------===//
//
// BlockPool manages a contiguous GPU slab that backs paged KV cache. The slab
// holds N physical blocks laid out as:
//
//   key_cache:   [num_blocks, block_size, kv_num_heads, head_dim]   (NHD)
//   value_cache: [num_blocks, block_size, kv_num_heads, head_dim]   (NHD)
//
// This layout matches the ORT com.microsoft.PagedAttention contrib op spec:
//   "Block-based key cache with shape (num_blocks, block_size, kv_num_heads,
//    head_size). This is updated in place within the op."
//
// Block allocation/free is host-side only (a simple free-list stack). The GPU
// sees a single contiguous allocation; the block_table tensor (managed by the
// ORT/OGA session scheduler) provides the logical→physical mapping at runtime.
//
// Both K and V are packed into one allocation (K first, then V) so a single
// hipHostMalloc covers the entire KV slab.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_RUNTIME_MM_BLOCK_POOL_H
#define HIPDNN_EP_RUNTIME_MM_BLOCK_POOL_H

#include "hal.h"

#include <cstddef>
#include <cstdint>

class BlockPool {
public:
  BlockPool() = default;
  ~BlockPool();

  // Non-copyable, non-movable (owns raw HAL allocations).
  BlockPool(const BlockPool &) = delete;
  BlockPool &operator=(const BlockPool &) = delete;

  // Allocate the physical KV slab via `hal`.
  // Returns false on allocation failure. After success, key_cache_base() and
  // value_cache_base() return valid GPU-accessible pointers.
  bool init(HalAllocator *hal, size_t num_blocks, int block_size,
            int kv_num_heads, int head_dim, int elem_size);

  // --- Host-side block allocation (O(1) free-list stack) ---

  // Returns a physical block index in [0, num_blocks()), or -1 if OOM.
  int alloc_block();

  // Returns `idx` to the free list. Caller must not use `idx` after this.
  void free_block(int idx);

  // --- Accessors ---

  // GPU-accessible base pointer for the key cache slab.
  void *key_cache_base() const { return kv_block_.gpu_ptr; }

  // GPU-accessible base pointer for the value cache slab.
  // V slab immediately follows K in the same allocation.
  void *value_cache_base() const {
    return static_cast<char *>(kv_block_.gpu_ptr) + v_offset_;
  }

  size_t num_blocks() const { return num_blocks_; }
  int num_free_blocks() const { return free_count_; }
  bool initialized() const { return kv_block_.gpu_ptr != nullptr; }

  // Per-block stride in bytes (= block_size * kv_num_heads * head_dim * elem).
  size_t block_stride_bytes() const { return block_stride_; }

  int block_size() const { return block_size_; }
  int kv_num_heads() const { return kv_num_heads_; }
  int head_dim() const { return head_dim_; }
  int elem_size() const { return elem_size_; }

private:
  HalBlock kv_block_ = {}; // GPU slab covering both K and V
  size_t v_offset_ = 0;    // byte offset from kv_block_.gpu_ptr to V
  size_t num_blocks_ = 0;
  size_t block_stride_ = 0; // bytes per physical block in K (or V) slab

  int block_size_ = 0;
  int kv_num_heads_ = 0;
  int head_dim_ = 0;
  int elem_size_ = 0;

  // Host-side free list: a simple stack of available block indices.
  // Allocated once with malloc, freed in destructor.
  int *free_list_ = nullptr;
  int free_count_ = 0;

  HalAllocator *hal_ = nullptr; // borrowed; not owned
};

#endif // HIPDNN_EP_RUNTIME_MM_BLOCK_POOL_H
