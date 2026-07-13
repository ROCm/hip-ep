/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- block_pool.cpp - Physical KV Cache Block Pool implementation -------===//

#include "block_pool.h"

#include <cstdlib>
#include <cstdio>

BlockPool::~BlockPool() {
  if (kv_block_.gpu_ptr && hal_) {
    hal_->free(kv_block_);
    kv_block_ = HalBlock{};
  }
  ::free(free_list_);
  free_list_ = nullptr;
  free_count_ = 0;
}

bool BlockPool::init(HalAllocator *hal, size_t num_blocks, int block_size,
                     int kv_num_heads, int head_dim, int elem_size) {
  if (num_blocks == 0 || block_size <= 0 || kv_num_heads <= 0 ||
      head_dim <= 0 || elem_size <= 0) {
    fprintf(stderr, "BlockPool::init: invalid parameters\n");
    return false;
  }

  hal_ = hal;
  num_blocks_ = num_blocks;
  block_size_ = block_size;
  kv_num_heads_ = kv_num_heads;
  head_dim_ = head_dim;
  elem_size_ = elem_size;

  // Each physical block in the K (or V) slab:
  //   [block_size, kv_num_heads, head_dim] × elem_size bytes
  block_stride_ =
      static_cast<size_t>(block_size) * kv_num_heads * head_dim * elem_size;

  // Total slab: K slab + V slab (same size each).
  size_t slab_each = num_blocks * block_stride_;
  size_t total = slab_each * 2;
  v_offset_ = slab_each;

  // Allocate via HAL so the correct hipHostMalloc flags are used (Mapped+NonCoherent
  // on APU for zero-copy GPU access; plain hipMalloc on discrete).
  kv_block_ = hal->alloc(total, MemTier::GPU);
  if (!kv_block_.gpu_ptr) {
    fprintf(stderr,
            "BlockPool::init: HAL alloc failed for %zu bytes "
            "(%zu blocks × %zu bytes K+V)\n",
            total, num_blocks, block_stride_ * 2);
    return false;
  }

  // Initialise the free list: push all block indices 0..num_blocks-1.
  free_list_ =
      static_cast<int *>(::malloc(sizeof(int) * num_blocks));
  if (!free_list_) {
    fprintf(stderr, "BlockPool::init: malloc for free_list failed\n");
    hal->free(kv_block_);
    kv_block_ = HalBlock{};
    return false;
  }
  for (size_t i = 0; i < num_blocks; ++i)
    free_list_[i] = static_cast<int>(i);
  free_count_ = static_cast<int>(num_blocks);

  return true;
}

int BlockPool::alloc_block() {
  if (free_count_ == 0)
    return -1;
  return free_list_[--free_count_];
}

void BlockPool::free_block(int idx) {
  if (idx < 0 || static_cast<size_t>(idx) >= num_blocks_) {
    fprintf(stderr, "BlockPool::free_block: out-of-range index %d\n", idx);
    return;
  }
  free_list_[free_count_++] = idx;
}
