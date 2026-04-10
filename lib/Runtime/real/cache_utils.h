/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_CACHE_UTILS_H
#define HIPDNN_EP_CACHE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <functional>

// Shared constants for runtime operator caches.
inline constexpr size_t kMaxWorkspaceBytes = 256ULL << 20; // 256 MB

// Boost-style hash combine: mixes `value` into `seed` using the golden-ratio
// constant. Used by all descriptor cache key hash structs.
inline void hash_combine(size_t &seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T> inline void hash_combine_val(size_t &seed, const T &v) {
  hash_combine(seed, std::hash<T>{}(v));
}

#endif // HIPDNN_EP_CACHE_UTILS_H
