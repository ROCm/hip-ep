/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace mm {

// Stream handle. Runtime consumers may pass a HIP stream pointer or nullptr.
using stream_t = void *;

using handle_t = std::uint64_t;

constexpr handle_t kInvalidHandle = 0;

enum class Status : std::int32_t {
  Ok = 0,
  ErrUnknown = 1,
  ErrInvalidArgument = 2,
  ErrAlreadyInit = 3,
  ErrNotInitialized = 4,
  ErrOutOfMemory = 5,
  ErrInvalidHandle = 6,
  ErrDoubleFree = 7,
  ErrHalFailure = 8
};

enum class MemoryClass : std::uint8_t {
  Generic = 0,
  Weight = 1,
  Activation = 2,
  KvCache = 3,
  Scratch = 4
};

enum class Lifetime : std::uint8_t {
  Static = 0,
  Request = 1,
  Step = 2,
  Transient = 3
};

enum class AccessPattern : std::uint8_t {
  Sequential = 0,
  Random = 1,
  WriteOnce = 2,
  ReadMostly = 3
};

// Single-GPU for now. Extend when multi-device support is implemented.
enum class Device : std::uint8_t {
  Gpu = 0,
};

enum class Tier : std::uint8_t { Hbm = 0, Dram = 1, Ssd = 2, Network = 3 };

struct AllocHints {
  MemoryClass mem_class = MemoryClass::Generic;
  Lifetime lifetime = Lifetime::Transient;
  std::size_t alignment = 0;
  AccessPattern access_pattern = AccessPattern::Sequential;
  Device device = Device::Gpu;
  bool shareable = false;
  std::size_t size_hint_max = 0;
};

struct AllocInfo {
  handle_t handle = kInvalidHandle;
  void *ptr = nullptr;
  std::size_t size = 0;
  MemoryClass mem_class = MemoryClass::Generic;
  Lifetime lifetime = Lifetime::Transient;
  int device = 0;
  Tier current_tier = Tier::Hbm;
  std::uint32_t ref_count = 1;
  AccessPattern access_pattern = AccessPattern::Sequential;
};

struct MetricsSnapshot {
  std::size_t total_allocated_bytes = 0;
  std::size_t peak_allocated_bytes = 0;
  std::uint64_t alloc_count = 0;
  std::uint64_t free_count = 0;
  std::uint64_t active_count = 0;
  std::size_t activation_bytes = 0;
  std::size_t activation_peak_bytes = 0;
  std::size_t kv_bytes = 0;
  std::size_t kv_peak_bytes = 0;
  std::uint64_t kv_alloc_count = 0;
  std::uint64_t kv_free_count = 0;
  std::uint64_t kv_block_handle_count = 0;
};

enum class KvFormat : std::uint8_t {
  Fp16 = 0,
  Fp8E4M3 = 1,
  Int4 = 2,
  TurboQuant4 = 3,
  TurboQuant3 = 4,
  TurboQuant2 = 5
};

struct KvBlockDesc {
  KvFormat format = KvFormat::Fp16;
  std::uint32_t block_size_tokens = 16;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t num_layers = 0;
  std::size_t bytes_per_token = 0;
  bool has_qjl_residual = false;
  std::uint32_t polar_bits = 0;
};

using kv_block_t = handle_t;

} // namespace mm
