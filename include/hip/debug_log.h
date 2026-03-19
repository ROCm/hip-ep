/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
  }();
  return enabled;
}

#define COMPILER_DEBUG_LOG(expr)                                               \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      llvm::errs() << expr;                                                    \
  } while (0)
