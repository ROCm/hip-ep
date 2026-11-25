/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

// Fingerprint host upload bytes outside all GPU event intervals. This is an
// equality checksum; preserved source/log/file bytes use SHA256 externally.
template <class T>
void gqaInputBinding(const char* shape, const char* name, const std::vector<T>& data) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
  const size_t count = data.size() * sizeof(T);
  for (size_t i = 0; i < count; ++i) {
    hash ^= bytes[i]; hash *= UINT64_C(1099511628211);
  }
  std::printf("UT_INPUT_BITS {\"shape\":\"%s\",\"buffer\":\"%s\",\"bytes\":%zu,"
              "\"fnv1a64\":\"%016llx\"}\n", shape, name, count,
              static_cast<unsigned long long>(hash));
}

// Validate the COPY test before generating references or indexing host KV.
// Zero splits requests automatic selection; positive values are explicit.
inline const char* gqaCopyInputError(int batch, int heads, int kv_heads,
    int dim, int capacity, int total, int window, int splits, int max_splits) {
  if (batch != 1 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0)
    return "COPY requires B=1 and positive H/G with H divisible by G";
  const int hpg = heads / kv_heads;
  if (dim != 128 || (hpg != 4 && hpg != 5 && hpg != 6 && hpg != 8))
    return "COPY requires D=128 and HpG in {4,5,6,8}";
  if (total < 2 || total > capacity)
    return "COPY requires 2 <= TOTAL <= MAX_SEQ";
  if (window > 0)
    return "COPY does not support a sliding window";
  if (splits < 0 || splits > max_splits)
    return "COPY SPLITS must be 0 (automatic) or within the supported limit";
  return nullptr;
}
