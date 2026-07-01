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

#ifdef _WIN32
// Static CRT (/MT) DLLs have their own CRT env — _dupenv_s can't see env vars
// set by the host process. Use Win32 API to read the real process environment.
extern "C" __declspec(
    dllimport) unsigned long __stdcall GetEnvironmentVariableA(const char *,
                                                               char *,
                                                               unsigned long);

namespace detail {
inline bool check_env(const char *name) {
  char buf[8];
  unsigned long n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  return n > 0 && buf[0] >= '1';
}
} // namespace detail
#endif

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_DEBUG");
#else
    const char *v = std::getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

// Sync-isolated profiling mode (HIPDNN_EP_PERF_ISOLATE=1). Inserts a
// hipStreamSynchronize at every OP_PROFILE scope boundary so each op's
// reported GPU time is its true standalone runtime, with no carry-over from
// prior queued work. Implies HIPDNN_EP_PERF=1. Kills concurrency by design;
// only useful as a diagnostic to find ops whose real cost is being masked by
// stream-queue depth in normal profiling.
inline bool hipdnn_ep_perf_isolate_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_PERF_ISOLATE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF_ISOLATE");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

// Convolution Find-algorithm cache toggle (A/B diagnostic). Default ON; set
// HIPDNN_EP_CONV_ALGO_CACHE=0 to disable, forcing a fresh
// miopenFindConvolutionForwardAlgorithm on every conv call. Latched once per
// process like the other flags.
inline bool hipdnn_ep_conv_algo_cache_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    char buf[8];
    unsigned long n =
        GetEnvironmentVariableA("HIPDNN_EP_CONV_ALGO_CACHE", buf, sizeof(buf));
    // Unset -> default ON; explicit "0" -> off.
    return !(n > 0 && buf[0] == '0');
#else
    const char *v = std::getenv("HIPDNN_EP_CONV_ALGO_CACHE");
    return !(v && v[0] == '0');
#endif
  }();
  return enabled;
}

inline bool hipdnn_ep_perf_enabled() {
  // PERF intentionally does NOT inherit from HIPDNN_EP_DEBUG: enabling PERF
  // forces a hipStreamSynchronize on every inference (so hipEventElapsedTime
  // can sample the H2D / Compute / D2H phases), which serializes the GPU
  // pipeline and skews measurements. Users who only want the per-call
  // [Runtime DEBUG] traces should not pay that cost.
  // ISOLATE implies PERF.
  static const bool enabled = [] {
#ifdef _WIN32
    if (detail::check_env("HIPDNN_EP_PERF"))
      return true;
    return detail::check_env("HIPDNN_EP_PERF_ISOLATE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF");
    if (v && v[0] >= '1')
      return true;
    const char *v2 = std::getenv("HIPDNN_EP_PERF_ISOLATE");
    return v2 && v2[0] >= '1';
#endif
  }();
  return enabled;
}

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)

// Conditional fprintf to stderr, gated on HIPDNN_EP_PERF only (DEBUG does
// not enable PERF; see hipdnn_ep_perf_enabled() above for why). Used by the
// per-inference [PERF] timing breakdown emitted from
// hipdnn_ep_runtime_tensor.cpp. Arguments are only evaluated when PERF is
// enabled, so leaving HIPDNN_EP_PERF unset has zero overhead.
#define RUNTIME_PERF_LOG(fmt, ...)                                             \
  do {                                                                         \
    if (hipdnn_ep_perf_enabled())                                              \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)
