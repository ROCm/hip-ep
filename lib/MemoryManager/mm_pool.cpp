/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * mm_pool.cpp — Static Pool allocator implementation.
 */

#include "mm/mm_pool.h"
#include "mm/mm_api.h"
#include <cstdlib>
#include <cstring>

struct mm_pool_s {
    void*       base;
    size_t      total_size;
    size_t*     offsets;
    uint32_t    num_entries;
    mm_handle_t handle;
};

mm_pool_t mm_pool_create(const mm_static_plan_t* plan) {
    if (!plan || plan->total_size == 0)
        return nullptr;

    mm_alloc_hints_t hints;
    hints.mem_class = MM_CLASS_WEIGHT;
    hints.lifetime  = MM_LIFETIME_STATIC;
    hints.alignment = 0;

    mm_handle_t h = mm_alloc(plan->total_size, &hints, nullptr);
    if (h == MM_HANDLE_INVALID)
        return nullptr;

    void* base = mm_get_ptr(h);
    if (!base) {
        mm_free(h, nullptr);
        return nullptr;
    }

    auto* pool = static_cast<mm_pool_s*>(std::malloc(sizeof(mm_pool_s)));
    if (!pool) {
        mm_free(h, nullptr);
        return nullptr;
    }

    pool->base       = base;
    pool->total_size  = plan->total_size;
    pool->handle      = h;
    pool->num_entries = plan->num_entries;
    pool->offsets     = nullptr;

    if (plan->num_entries > 0 && plan->offsets) {
        size_t offsets_bytes = sizeof(size_t) * plan->num_entries;
        pool->offsets = static_cast<size_t*>(std::malloc(offsets_bytes));
        if (!pool->offsets) {
            mm_free(h, nullptr);
            std::free(pool);
            return nullptr;
        }
        std::memcpy(pool->offsets, plan->offsets, offsets_bytes);
    }

    return pool;
}

void* mm_pool_get_ptr(mm_pool_t pool, uint32_t index) {
    if (!pool || index >= pool->num_entries)
        return nullptr;
    return static_cast<char*>(pool->base) + pool->offsets[index];
}

void* mm_pool_get_base(mm_pool_t pool) {
    return pool ? pool->base : nullptr;
}

size_t mm_pool_get_size(mm_pool_t pool) {
    return pool ? pool->total_size : 0;
}

uint32_t mm_pool_get_num_entries(mm_pool_t pool) {
    return pool ? pool->num_entries : 0;
}

void mm_pool_destroy(mm_pool_t pool) {
    if (!pool)
        return;
    if (pool->handle != MM_HANDLE_INVALID)
        mm_free(pool->handle, nullptr);
    std::free(pool->offsets);
    std::free(pool);
}
