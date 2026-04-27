/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
// Set HIPDNN_EP_PERF=1 to enable [PERF] aggregate (wall + GPU breakdown).
// Set HIPDNN_EP_PERF_EACH=1 to additionally emit per-inference [PERF] rows.
// Set HIPDNN_EP_PERF_WARMUP=N to bucket the first N inferences of each
//   session as "warmup" in BOTH the [PERF] and [OP_PROFILE] aggregates so
//   their kernel-selection / JIT / autotune cost does not pollute the
//   steady-state averages. Default: 1. Set 0 to disable the split.
#include <cstdio>
#include <cstdlib>

struct RuntimeState;

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
  }();
  return enabled;
}

inline bool hipdnn_ep_perf_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_PERF");
    return v && v[0] >= '1';
  }();
  return enabled || hipdnn_ep_debug_enabled();
}

// Number of leading inferences (per session) to bucket as "warmup". The
// first call into the EP triggers all kinds of one-time work -- hipBLASLt
// heuristic search, MIOpen kernel finder cache fills, hipRTC JIT, the
// first hipMalloc growth, runtime constant H2D, etc. -- that is irrelevant
// to steady-state behavior and skews avg/p50 if mixed in. We still RECORD
// these inferences so the user can see exactly how expensive warmup was;
// they just don't contribute to the steady-state numbers.
//
// Shared by both PERF (hipdnn_ep_runtime_tensor.cpp) and OP_PROFILE
// (operator_profile.cpp) so the two aggregates partition identically and
// their averages can be compared directly.
inline unsigned hipdnn_ep_perf_warmup_count() {
  static const unsigned n = [] {
    const char *v = std::getenv("HIPDNN_EP_PERF_WARMUP");
    if (!v || !*v)
      return 1u;
    int parsed = std::atoi(v);
    return parsed < 0 ? 0u : static_cast<unsigned>(parsed);
  }();
  return n;
}

// Implemented in hipdnn_ep_runtime_tensor.cpp. Drives the final inference
// rollup (using `state`'s stream to record the closing d2h_end marker) and
// prints the aggregate PERF summary. Safe to call when PERF is disabled.
extern "C" void hipdnn_ep_perf_flush_and_print(RuntimeState *state);

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)
