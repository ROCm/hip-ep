/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
// Set HIPDNN_EP_PERF=1 to enable only [PERF] timing breakdown per inference.
#include <cstdio>
#include <cstdlib>

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
    const char *v = getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
  }();
  return enabled;
}

inline bool hipdnn_ep_perf_enabled() {
  static const bool enabled = [] {
    const char *v = getenv("HIPDNN_EP_PERF");
    return v && v[0] >= '1';
  }();
  return enabled || hipdnn_ep_debug_enabled();
}

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)
