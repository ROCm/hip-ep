/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Lightweight CPU phase-timing utilities for session initialisation and
// compilation diagnostics.  Pure C++ stdlib -- no LLVM, HIP, or external deps
// so the header can be shared across the compiler stack, the bitcode-compiled
// runtime, the MLIR backend plugins, and standalone tools.
//
// Design rationale:
//   * Free functions, not classes -- keeps measurement separate from output
//     (callers choose fprintf, llvm::errs(), std::cout, etc.).
//   * Evaluated against similar timing utilities in upstream projects.
//     A full TimingManager would be overkill for timing a linear sequence
//     of coarse init phases.

#pragma once
#include <chrono>
#include <cstdlib>

// Returns true if HIPDNN_EP_TIMING env var is set to >= "1".
// Result is cached on first call (thread-safe per C++11 [stmt.dcl]/4).
inline bool hipdnn_ep_timing_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_TIMING");
    return v && v[0] >= '1';
  }();
  return enabled;
}

// Records the current time, computes seconds elapsed since `marker`,
// then resets `marker` to the current time.  Use for sequential phases.
inline double record_elapsed(std::chrono::steady_clock::time_point &marker) {
  auto now = std::chrono::steady_clock::now();
  double s = std::chrono::duration<double>(now - marker).count();
  marker = now;
  return s;
}

// Returns seconds elapsed since `marker` without resetting it.
// Use for total-elapsed measurements.
inline double elapsed_since(std::chrono::steady_clock::time_point marker) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       marker)
      .count();
}
