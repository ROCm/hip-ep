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

#include "hip/env.h" // single cross-platform env reader (see its header)

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = hipdnn_ep::env_enabled("HIPDNN_EP_DEBUG");
  return enabled;
}

// W4A8 integer-dot-product (dp4a) path for matmul_nbits GEMV. When enabled,
// eligible bits==4, K%32==0 fp16 GEMVs dynamically quantize the activation to
// per-group int8 and use a v_dot4_i32_iu8 dot product instead of the
// dequant-ALU-bound fp GEMV.
// DEFAULT-ON (so CI validates the optimization); set HIPDNN_EP_MATMUL_DP4A=0
// to force the classic fp GEMV path for A/B isolation. Latched on first read
// like the other flags here.
inline bool hipdnn_ep_matmul_dp4a_enabled() {
  static const bool enabled =
      hipdnn_ep::env_enabled_default_on("HIPDNN_EP_MATMUL_DP4A");
  return enabled;
}

// Widest M the dp4a path claims. 1 is decode; above that is the speculative
// verify pass, where blocking rows over one pass of the weight is the whole
// point (see the GEMV-M dp4a section in matmul_nbits_kernel.hip). The default
// stops below the fp path's WMMA cutoff, which owns M >= 8 and is measured
// per shape there. HIPDNN_EP_MATMUL_DP4A_MAX_M=1 restores decode-only
// behaviour, which is the A/B control for the verify-pass claim.
inline int hipdnn_ep_matmul_dp4a_max_m() {
  static const int m = [] {
    const std::string v = hipdnn_ep::env_string("HIPDNN_EP_MATMUL_DP4A_MAX_M");
    if (v.empty())
      return 7;
    const int parsed = atoi(v.c_str());
    return (parsed >= 1 && parsed <= 64) ? parsed : 7;
  }();
  return m;
}

inline bool hipdnn_ep_perf_enabled() {
  // PERF intentionally does NOT inherit from HIPDNN_EP_DEBUG: enabling PERF
  // forces a hipStreamSynchronize on every inference (so hipEventElapsedTime
  // can sample the H2D / Compute / D2H phases), which serializes the GPU
  // pipeline and skews measurements. Users who only want the per-call
  // [Runtime DEBUG] traces should not pay that cost.
  // A set HIPDNN_EP_TRACE_FILE also implies PERF (the trace needs the profiler
  // running).
  static const bool enabled =
      hipdnn_ep::env_enabled("HIPDNN_EP_PERF") ||
      !hipdnn_ep::env_string("HIPDNN_EP_TRACE_FILE").empty();
  return enabled;
}

// Chrome-trace output path (HIPDNN_EP_TRACE_FILE). Empty => tracing disabled.
// A non-empty path writes a chrome://tracing JSON of the per-op timeline (one
// fenceless marker per op, GPU time derived by differencing consecutive
// markers) and also flips hipdnn_ep_perf_enabled() on.
inline const std::string &hipdnn_ep_trace_path() {
  static const std::string path = hipdnn_ep::env_string("HIPDNN_EP_TRACE_FILE");
  return path;
}
inline bool hipdnn_ep_trace_enabled() {
  return !hipdnn_ep_trace_path().empty();
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
