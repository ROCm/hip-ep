/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
// Set HIPDNN_EP_PERF=1 to enable only [PERF] timing breakdown per inference.
//
// On Windows, model DLLs link static CRT (libcmt.lib), so std::getenv()
// can't see environment variables set by the host process.
// GetEnvironmentVariableA reads the shared process environment block.
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
extern "C" unsigned long __stdcall GetEnvironmentVariableA(const char *,
                                                           char *,
                                                           unsigned long);
#endif

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    char buf[8] = {};
    unsigned long n =
        GetEnvironmentVariableA("HIPDNN_EP_DEBUG", buf, sizeof(buf));
    return n > 0 && buf[0] >= '1';
#else
    const char *v = std::getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

inline bool hipdnn_ep_perf_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    char buf[8] = {};
    unsigned long n =
        GetEnvironmentVariableA("HIPDNN_EP_PERF", buf, sizeof(buf));
    return n > 0 && buf[0] >= '1';
#else
    const char *v = std::getenv("HIPDNN_EP_PERF");
    return v && v[0] >= '1';
#endif
  }();
  return enabled || hipdnn_ep_debug_enabled();
}

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)
