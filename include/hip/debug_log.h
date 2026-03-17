/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Compiler-side debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all
// [ONNX→HIP]/[GenerateInterface]/[HipToLLVM] output.
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
    const char *v = getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
  }();
  return enabled;
}

// Usage: COMPILER_DEBUG_LOG("[TAG] text " << value << "\n");
#define COMPILER_DEBUG_LOG(expr)                                               \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      llvm::errs() << expr;                                                    \
  } while (0)

// Usage: DRIVER_DEBUG_LOG("[TAG] text " << value << "\n");
#define DRIVER_DEBUG_LOG(expr)                                                 \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      std::cerr << expr;                                                       \
  } while (0)
