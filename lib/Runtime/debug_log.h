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

// ============================================================================
// Dynamic-shape debug tracers (added with the data-dependent dynamic output
// shapes feature). All three are zero-overhead when off: each gate is a
// `static const bool` checked once at first use, then branch-predicted.
//
// HIPDNN_EP_DEBUG_SHAPES=1
//     Dumps a one-line summary on the EP side for every dynamic output dim
//     resolved at pre- and post-Compute() phases:
//       [Shapes pre]  out[0] = [16, ?, ?]  spec[1]=RuntimeSlot(2) ...
//       [Shapes post] out[0] = [16, 7, 8]  via slots {2: 7, 3: 8}
//     Use this when you suspect ComposeDimSpecs is computing the wrong shape
//     for an output but the kernels themselves are fine.
//
// HIPDNN_EP_TRACE_SLOTS=1
//     Traces every publish_dim / read_dim / publish_buffer / read_buffer call
//     inside the model.dll:
//       [Slots] publish_dim(7) = 12     <- wrap_nonzero
//       [Slots] read_dim(7)    = 12     <- wrap_shape consumer
//     Use this when you need to confirm that a producer fires before its
//     consumer, or to localize a read-before-publish abort.
//
// HIPDNN_EP_VALIDATE_SHAPES=1
//     INTENTIONALLY NOT WIRED -- gate exists, no call site reads it. Setting
//     this is currently a no-op. Designed-but-deferred: after every
//     Compute() that produces a dynamic-shape output, would run the same
//     subgraph on the ORT CPU EP and compare **shape only** (not values),
//     diverging with LOG(ERROR) on mismatch. Deferred because the existing
//     numeric suite (test/numeric/) already catches DimSpec / resolver bugs
//     as fatal aborts or numeric divergence, and the framework cost of
//     building, holding, and per-Compute()-running a second ORT session per
//     fused MlirCustomOp is high relative to the residual coverage gain.
//     See docs/design/dynamic-shape-debug-surface.md (`Deferred` section)
//     for the full rationale, the slot-in point, and the trigger conditions
//     for reviving it (new DimSpec node kinds, a real silent-shape-divergence
//     bug, or post-DimSpec-composition compiler optimizations).
// ============================================================================

inline bool hipdnn_ep_debug_shapes_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_DEBUG_SHAPES");
#else
    const char *v = std::getenv("HIPDNN_EP_DEBUG_SHAPES");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

inline bool hipdnn_ep_trace_slots_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_TRACE_SLOTS");
#else
    const char *v = std::getenv("HIPDNN_EP_TRACE_SLOTS");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

inline bool hipdnn_ep_validate_shapes_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_VALIDATE_SHAPES");
#else
    const char *v = std::getenv("HIPDNN_EP_VALIDATE_SHAPES");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      fprintf(stderr, fmt, ##__VA_ARGS__);                                     \
  } while (0)

// Zero-overhead trace point for the dynamic-output slot ABI. Gated on
// HIPDNN_EP_TRACE_SLOTS=1; argument-evaluation cost is zero when off.
#define HIPDNN_EP_SLOT_TRACE(fmt, ...)                                         \
  do {                                                                         \
    if (hipdnn_ep_trace_slots_enabled())                                       \
      fprintf(stderr, "[Slots] " fmt "\n", ##__VA_ARGS__);                     \
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
