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
#include <cstring>

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

// Full string read for comma-separated op lists (see
// HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS). Uses Win32 so /MT model.dll sees the host
// process environment.
inline const char *read_env_string_into(char *buf, unsigned buf_size,
                                        const char *name, unsigned long *out_len) {
  unsigned long n = GetEnvironmentVariableA(name, buf, buf_size);
  *out_len = n;
  if (n == 0 || n >= buf_size)
    return nullptr;
  buf[n] = '\0';
  return buf;
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

// HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS: comma-separated ONNX op names (e.g.
// "Gather,Where") or "*" / "all" for every op. Case-insensitive whole-token
// match. Parsed once per process.
inline bool hipdnn_ep_debug_cpu_fallback_ops_contains(const char *token) {
  static bool parsed = false;
  static bool wildcard = false;
  static char tokens[64][48];
  static int token_count = 0;

  auto fold_eq = [](const char *a, const char *b) {
    for (; *a && *b; ++a, ++b) {
      char ca = *a;
      char cb = *b;
      if (ca >= 'A' && ca <= 'Z')
        ca = static_cast<char>(ca - 'A' + 'a');
      if (cb >= 'A' && cb <= 'Z')
        cb = static_cast<char>(cb - 'A' + 'a');
      if (ca != cb)
        return false;
    }
    return *a == '\0' && *b == '\0';
  };

  if (!parsed) {
    parsed = true;
    static char storage[2048];
    const char *list = nullptr;
#ifdef _WIN32
    unsigned long n = 0;
    list = detail::read_env_string_into(storage, sizeof(storage),
                                       "HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS", &n);
#else
    list = std::getenv("HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS");
#endif
    if (!list || !list[0])
      return false;
    const char *p = list;
    while (*p && token_count < 64) {
      while (*p == ',' || *p == ' ' || *p == '\t')
        ++p;
      if (!*p)
        break;
      const char *start = p;
      while (*p && *p != ',')
        ++p;
      const char *end = p;
      while (end > start && (end[-1] == ' ' || end[-1] == '\t'))
        --end;
      const size_t seg_len = static_cast<size_t>(end - start);
      if (seg_len > 0 && seg_len < sizeof(tokens[0])) {
        for (size_t i = 0; i < seg_len; ++i)
          tokens[token_count][i] = start[i];
        tokens[token_count][seg_len] = '\0';
        if (fold_eq(tokens[token_count], "*") ||
            fold_eq(tokens[token_count], "all")) {
          wildcard = true;
        }
        ++token_count;
      }
      if (*p == ',')
        ++p;
    }
  }

  if (wildcard)
    return true;
  if (!token || !token[0])
    return false;
  for (int i = 0; i < token_count; ++i) {
    if (fold_eq(tokens[i], token))
      return true;
  }
  return false;
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
