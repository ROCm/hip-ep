/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MM_BLOCK_TABLE_H
#define MM_BLOCK_TABLE_H

#include "mm_hal.h"
#include "mm_types.h"

#include <cstdint>
#include <vector>

namespace mm {

/*
 * Block table: maps logical block indices to physical GPU pointers.
 *
 * Used by paged attention kernels to look up KV data scattered
 * across non-contiguous physical blocks.
 *
 * Layout: block_table[i] = GPU pointer for logical block i.
 */
class BlockTable {
public:
  /* Append a block's GPU pointer to the table. */
  void add_block(void *gpu_ptr);

  /* Get the pointer array (for passing to kernels). */
  void **data();
  const void *const *data() const;

  /* Number of entries. */
  uint32_t size() const;

  /* Clear the table. */
  void clear();

  /*
   * Materialize all blocks into a contiguous buffer.
   * Copies block_size_bytes from each block pointer into dst
   * sequentially: dst[0..block_size), dst[block_size..2*block_size), ...
   *
   * Used as a bridge for existing GEMM kernels that expect contiguous KV.
   * Returns 0 on success.
   */
  int materialize_contiguous(void *dst, size_t block_size_bytes,
                             const mm_hal_t *hal, mm_stream_t stream) const;

private:
  std::vector<void *> ptrs_;
};

} // namespace mm

#endif /* MM_BLOCK_TABLE_H */
