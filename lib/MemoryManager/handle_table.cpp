/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "handle_table.h"

mm_handle_t HandleTable::insert(void* ptr, size_t size, mm_class_t mem_class,
                                mm_lifetime_t lifetime, mm_device_t device) {
    std::lock_guard<std::mutex> lock(mutex_);
    mm_handle_t handle = next_handle_++;

    mm_alloc_info_t info;
    info.handle   = handle;
    info.ptr      = ptr;
    info.size     = size;
    info.mem_class = mem_class;
    info.lifetime  = lifetime;
    info.device    = device;

    table_[handle] = info;
    return handle;
}

bool HandleTable::lookup(mm_handle_t handle, mm_alloc_info_t* info) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = table_.find(handle);
    if (it == table_.end())
        return false;
    if (info)
        *info = it->second;
    return true;
}

bool HandleTable::remove(mm_handle_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.erase(handle) > 0;
}

size_t HandleTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.size();
}

void HandleTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
}
