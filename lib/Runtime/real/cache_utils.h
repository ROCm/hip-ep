/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_CACHE_UTILS_H
#define HIPDNN_EP_CACHE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <functional>

// Boost-style hash combine: mixes `value` into `seed` using the golden-ratio
// constant.
inline void hash_combine(size_t &seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T> inline void hash_combine_val(size_t &seed, const T &v) {
  hash_combine(seed, std::hash<T>{}(v));
}

#endif // HIPDNN_EP_CACHE_UTILS_H
