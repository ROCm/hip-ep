/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "mm/mm_types.h"

namespace mm {

struct WatermarkConfig {
  // 80% is a standard "yellow zone" threshold — enough headroom to start
  // eviction / spilling without risking OOM on the next large allocation.
  double warn = 0.80;
  // 95% leaves only a thin margin; at this point synchronous blocking
  // (stall the pipeline until memory is freed) is preferable to an OOM crash.
  double critical = 0.95;
};

struct TierConfig {
  bool enable_cpu = false;
  bool enable_ssd = false;
  bool enable_cxl = false;
};

struct Config {
  int device_id = 0;
  // 256B alignment matches HIP's minimum guaranteed alignment and avoids
  // unaligned-access penalties on RDNA3 cache lines.
  std::size_t default_alignment = 256;
  std::size_t gpu_memory_limit = 0; // 0 = query from HAL.
  // 90% to KV: LLM inference is dominated by KV cache memory; leaving 10%
  // for activations/scratch is sufficient since those are short-lived and
  // recycled by the arena.
  double kv_cache_fraction = 0.90;
  std::size_t kv_cache_max_bytes =
      0; // Explicit cap for KV pool. 0 = derive from fraction.
  // 16 tokens/block balances granularity (waste ≤ 15 tokens per sequence)
  // against per-block bookkeeping overhead. Matches vLLM's default.
  std::uint32_t kv_block_size_tokens = 16;
  std::size_t kv_bytes_per_token_hint =
      0; // Optional override when descriptor omits bytes/token.
  // Power-of-4 progression from 1 KB to 4 MB covers the typical activation
  // tensor range (small per-token buffers to full-layer intermediates).
  // The last class is a catch-all for oversized allocations.
  std::array<std::size_t, 8> activation_size_class_upper_bounds = {
      1024,
      4 * 1024,
      16 * 1024,
      64 * 1024,
      256 * 1024,
      1024 * 1024,
      4 * 1024 * 1024,
      std::numeric_limits<std::size_t>::max()};
  // 256 KB slab: large enough to hold several typical activations per class
  // without over-committing GPU memory across 8 classes (8 × 256 KB = 2 MB).
  std::size_t activation_slab_bytes = 256 * 1024;
  bool enable_debug_log = false;
  bool enable_metrics = true;
  WatermarkConfig watermarks{};
  TierConfig tiers{};
};

Config config_default();

} // namespace mm
