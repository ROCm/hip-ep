/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/*
 * KV cache public API.
 */

#pragma once

#include <cstddef>

#include "mm/mm_types.h"

namespace mm {

kv_block_t kv_alloc_block(const KvBlockDesc &desc);
Status kv_free_block(kv_block_t block);
kv_block_t kv_fork_block(kv_block_t source);
bool kv_get_block_table(const kv_block_t *blocks, std::size_t count,
                        void **out_ptrs);
bool kv_get_block_desc(kv_block_t block, KvBlockDesc *out_desc);

} // namespace mm
