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
#include <string>

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

// Low-distortion per-op profiling: instead of a start+stop hipEvent PAIR per
// op (two system-fenced records that the comment in op_profile.cpp notes
// "dominated the per-op table"), record a SINGLE fenceless marker event at the
// end of each op and derive each op's GPU time by differencing consecutive
// markers on the (in-order) stream. Halves the record count, drops the
// per-record system fence, and never stream-syncs per op -- so the measurement
// perturbs the run far less than the classic pair mode. Opt in with
// HIPDNN_EP_PERF_TIMELINE=1 (implies HIPDNN_EP_PERF=1). Mutually exclusive with
// ISOLATE (which deliberately serializes for standalone timings).
inline bool hipdnn_ep_perf_timeline_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_PERF_TIMELINE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF_TIMELINE");
    return v && v[0] >= '1';
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
  // ISOLATE and TIMELINE both imply PERF.
  static const bool enabled = [] {
#ifdef _WIN32
    if (detail::check_env("HIPDNN_EP_PERF"))
      return true;
    if (detail::check_env("HIPDNN_EP_PERF_ISOLATE"))
      return true;
    return detail::check_env("HIPDNN_EP_PERF_TIMELINE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF");
    if (v && v[0] >= '1')
      return true;
    const char *v2 = std::getenv("HIPDNN_EP_PERF_ISOLATE");
    if (v2 && v2[0] >= '1')
      return true;
    const char *v3 = std::getenv("HIPDNN_EP_PERF_TIMELINE");
    return v3 && v3[0] >= '1';
#endif
  }();
  return enabled;
}

// Chrome-trace output path (HIPDNN_EP_TRACE_FILE). Empty => tracing disabled.
// When set (and HIPDNN_EP_PERF/TIMELINE is on), op_profile emits a per-op
// chrome://tracing JSON in addition to the aggregated [PERF] table.
inline const std::string &hipdnn_ep_trace_path() {
  static const std::string path = [] {
#ifdef _WIN32
    char buf[1024];
    unsigned long n =
        GetEnvironmentVariableA("HIPDNN_EP_TRACE_FILE", buf, sizeof(buf));
    return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
    const char *v = std::getenv("HIPDNN_EP_TRACE_FILE");
    return v ? std::string(v) : std::string();
#endif
  }();
  return path;
}
inline bool hipdnn_ep_trace_enabled() { return !hipdnn_ep_trace_path().empty(); }

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
