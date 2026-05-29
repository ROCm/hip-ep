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

struct Config {
  int device_id = 0;
  std::size_t default_alignment = 256;
  std::uint32_t kv_block_size_tokens = 16;
  std::size_t kv_bytes_per_token_hint = 0;
  std::array<std::size_t, 8> activation_size_class_upper_bounds = {
      1024,
      4 * 1024,
      16 * 1024,
      64 * 1024,
      256 * 1024,
      1024 * 1024,
      4 * 1024 * 1024,
      std::numeric_limits<std::size_t>::max()};
  std::size_t activation_slab_bytes = 256 * 1024;
  bool enable_debug_log = false;
  bool enable_metrics = true;
};

Config config_default();

} // namespace mm
