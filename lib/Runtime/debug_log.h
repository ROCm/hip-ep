/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
// Set HIPDNN_EP_PERF=1 to enable only [PERF] timing breakdown per inference.
#include <cstdio>
#include <string>

#include "hip/env.h" // single cross-platform env reader (see its header)

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = hipdnn_ep::env_enabled("HIPDNN_EP_DEBUG");
  return enabled;
}

// W4A8 integer-dot-product (dp4a) path for matmul_nbits single-row (M==1)
// decode GEMV. When enabled, eligible bits==4, K%32==0 fp16 decode GEMVs
// dynamically quantize the activation to per-group int8 and use a
// v_dot4_i32_iu8 dot product instead of the dequant-ALU-bound fp GEMV.
// DEFAULT-ON (so CI validates the optimization); set HIPDNN_EP_MATMUL_DP4A=0
// to force the classic fp GEMV path for A/B isolation. Latched on first read
// like the other flags here.
inline bool hipdnn_ep_matmul_dp4a_enabled() {
  static const bool enabled =
      hipdnn_ep::env_enabled_default_on("HIPDNN_EP_MATMUL_DP4A");
  return enabled;
}

// Minimum expert count for the grouped WMMA MoE prefill to be worthwhile.
// Grouped amortizes one launch across all experts; it only beats the per-expert
// host loop when there are many experts with small per-expert GEMMs. Measured:
// gpt-oss-20b (32 experts) is ~18% faster on the host loop at P128 and P2048,
// so few-expert models fall back. Default 64 sits between gpt-oss (32) and the
// many-expert models (120b=128, A3B=256). CI-tunable via
// HIPDNN_EP_QMOE_GROUPED_MIN_EXPERTS to sweep the crossover.
inline int hipdnn_ep_qmoe_grouped_min_experts() {
  static const int v =
      hipdnn_ep::env_int("HIPDNN_EP_QMOE_GROUPED_MIN_EXPERTS", 64);
  return v;
}

// Fill-aware post-bucket gate for grouped WMMA MoE prefill.
//
// The num_experts gate above is a poor predictor: grouped WMMA only wins when
// each active expert fills its 64-row GEMM tile. gpt-oss-120b has 128 experts
// (passes the count gate) but with k=4 routing its per-expert fill is ~7.6 rows
// (P128) so the 64-row tile is mostly padding and the launch is gated by the
// heaviest expert (tail) -> +13-37% TTFT regression. Qwen3.6-35B-A3B (256
// experts, k=8) fills the tile well and wins. The separating signal is the
// actual per-expert row distribution (h_counts), known only after bucketing.
//
// Decision (all must hold, else fall back to the host loop = pre-existing main
// behavior, so there is no regression vs main):
//   num_active >= MIN_ACTIVE  AND  dense_row_fraction >= MIN_DENSE_FRAC
// where an expert is "dense" if its routed row count >= TILE_ROWS, and
//   dense_row_fraction = (rows in dense experts) / (num_tokens * k).

// Per-expert row count at/above which an expert fills the WMMA GEMM tile.
inline int hipdnn_ep_qmoe_grouped_tile_rows() {
  static const int v =
      hipdnn_ep::env_int("HIPDNN_EP_QMOE_GROUPED_TILE_ROWS", 64);
  return v;
}

// Minimum number of active experts for grouped's single-launch amortization to
// beat the host loop. Few-expert models (gpt-oss-20b, <=32 active) issue few,
// large, efficient per-expert GEMMs and lose with grouped even when dense.
inline int hipdnn_ep_qmoe_grouped_min_active() {
  static const int v =
      hipdnn_ep::env_int("HIPDNN_EP_QMOE_GROUPED_MIN_ACTIVE", 64);
  return v;
}

// Minimum fraction (percent, 0-100) of routed rows that must land in dense
// experts for grouped WMMA to be selected. Conservative default: when in doubt,
// use the host loop. Percent (not float) to reuse the int env parser.
inline int hipdnn_ep_qmoe_grouped_min_dense_pct() {
  static const int v =
      hipdnn_ep::env_int("HIPDNN_EP_QMOE_GROUPED_MIN_DENSE_PCT", 70);
  return v;
}

// Minimum prefill length (num_tokens) for grouped WMMA MoE prefill to be
// selected. The dense-fraction gate above is scale-free (a ratio), so it admits
// grouped for layers whose *shape* fills the tile even when the *total* prefill
// is too small to amortize grouped's per-expert launch overhead. Measured on
// Qwen3.6-35B-A3B (VLM image+text prefill), grouped vs host-loop TTFT:
//   1842 tok: +4.7%   1949 tok: +5.9%   (grouped LOSES)   <- default vlm CSV pt
//   2224 tok: -3.3%   2540 tok: -5.7%   3227 tok: -4.8%   3952 tok: -18.9%
// i.e. a clean crossover at ~2.0-2.1k tokens. Below it, fall back to the host
// loop (= pre-existing main behavior, so no regression vs main); above it keep
// grouped and its large long-context win. Default 2048 sits in the
// verified-safe window (1949, 2224]. CI-tunable via
// HIPDNN_EP_QMOE_GROUPED_MIN_TOKENS.
inline int hipdnn_ep_qmoe_grouped_min_tokens() {
  static const int v =
      hipdnn_ep::env_int("HIPDNN_EP_QMOE_GROUPED_MIN_TOKENS", 2048);
  return v;
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
