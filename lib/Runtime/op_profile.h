/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Per-operator GPU profiling (gated on HIPDNN_EP_PERF env var).
//
// Usage: add OP_PROFILE("opname", shape_lambda, state) at the top of each
// runtime wrapper. The shape_lambda is a callable returning std::string,
// invoked only when profiling is active — zero overhead when disabled.
//
// Timing model (single, low-distortion): per inference one "epoch" event is
// recorded on the stream, then one fenceless marker event per op. Each op's GPU
// time is the gap between its marker and the previous one on the (in-order)
// stream, resolved in bulk after the per-inference stream sync -- one fenceless
// record per op, no per-op start/stop pair and no per-op stream sync.

#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include <chrono>
#include <optional>
#include <string>

struct OpProfileState;

OpProfileState *op_profile_create();
void op_profile_destroy(OpProfileState *ps);
void op_profile_reset(OpProfileState *ps);
void op_profile_resolve_and_print(OpProfileState *ps);
void op_profile_add_pending(OpProfileState *ps, const std::string &name,
                            const std::string &shape, int markerIndex,
                            double cpuMs, int64_t bytes = 0,
                            double cpuStartUs = 0.0);
// Add H2D/Compute/D2H pipeline spans (ms) for the most recently resolved
// inference to the chrome trace's dedicated tracks. No-op unless tracing is on.
void op_profile_add_io_spans(OpProfileState *ps, double h2dMs, int64_t h2dBytes,
                             double computeMs, double d2hMs, int64_t d2hBytes);
// Record the per-inference epoch event (once, on the first op) plus acquire a
// fenceless marker per op; per-op GPU time is derived by differencing markers
// against the epoch on the in-order stream.
void op_profile_ensure_epoch(OpProfileState *ps, hipStream_t stream);
int op_profile_acquire_marker(OpProfileState *ps);
hipEvent_t op_profile_get_marker_event(OpProfileState *ps, int index);
// Memory roofline peak (GB/s) used for the chrome trace's %peak arg. Currently
// a hardcoded constant (see op_profile.cpp).
double op_profile_roofline_gbps();
void op_profile_add_cpu(OpProfileState *ps, const std::string &name,
                        double cpuMs);
void op_profile_add_cpu_total(OpProfileState *ps, const std::string &name,
                              double cpuStartUs, double cpuMs);
bool op_profile_is_active(OpProfileState *ps);

// One-shot RGP capture fence. Gated on the RGP_FENCE env var; a no-op (single
// cached check) when unset, so it costs nothing on normal runs. When RGP_FENCE
// names this op, the fence drains the GPU (hipDeviceSynchronize) and sleeps
// RGP_FENCE_MS ms to manufacture an idle window, so an RGP dispatch-mode
// auto-capture arms deterministically on THIS op's first dispatch after the gap
// (dispatch-index positioning is unreliable in this build). RGP_FENCE_SKIP=N
// arms on the (N+1)-th matching instance instead of the first (e.g. to target a
// full-attention layer rather than layer-0 sliding attention). Fires once per
// process and emits a "[RGP_FENCE_ARMED]" stderr marker the orchestrator waits
// on before triggering the capture.
void rgp_capture_fence(const char *opname);

// Absolute microseconds on the shared steady_clock axis. A plain time
// conversion tied to no session: the trace axis is process-global, so all
// sessions and inferences land on it without any captured baseline.
inline double
op_profile_us_since_epoch(std::chrono::steady_clock::time_point tp) {
  return std::chrono::duration<double, std::micro>(tp.time_since_epoch())
      .count();
}

struct OpProfileCpuScope {
  OpProfileState *ps;
  std::string name;
  std::chrono::steady_clock::time_point cpuStart;

  OpProfileCpuScope(OpProfileState *p, std::string n)
      : ps(p), name(std::move(n)), cpuStart(std::chrono::steady_clock::now()) {}

  ~OpProfileCpuScope() {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cpuStart)
                    .count();
    if (ps)
      op_profile_add_cpu(ps, name, ms);
  }

  OpProfileCpuScope(const OpProfileCpuScope &) = delete;
  OpProfileCpuScope &operator=(const OpProfileCpuScope &) = delete;
};

struct OpProfileScope {
  OpProfileState *ps;
  std::string name;
  std::string shape;
  int markerIndex;
  hipStream_t stream;
  int64_t bytes;
  double cpuTsUs = 0.0; // op start, absolute us on the shared trace axis
  std::chrono::steady_clock::time_point cpuStart;

  OpProfileScope(OpProfileState *p, std::string n, std::string sh,
                 hipStream_t s, int mkIdx, int64_t b = 0)
      : ps(p), name(std::move(n)), shape(std::move(sh)), markerIndex(mkIdx),
        stream(s), bytes(b) {
    cpuStart = std::chrono::steady_clock::now();
    cpuTsUs = op_profile_us_since_epoch(cpuStart);
    // The epoch is recorded by the macro before this op's kernels are enqueued;
    // the op's GPU time is derived from the marker recorded in the destructor.
    // Nothing to enqueue here -- no start event, no stream sync.
  }

  ~OpProfileScope() {
    // Single fenceless marker at the end of the op; no stream sync. Per-op
    // GPU duration = this marker minus the previous one (resolved at print).
    (void)hipEventRecord(op_profile_get_marker_event(ps, markerIndex), stream);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cpuStart)
                    .count();
    if (ps)
      op_profile_add_pending(ps, name, shape, markerIndex, ms, bytes, cpuTsUs);
  }

  OpProfileScope(const OpProfileScope &) = delete;
  OpProfileScope &operator=(const OpProfileScope &) = delete;
};

#define OP_PROFILE(opname, shape_fn, state_arg)                                \
  OP_PROFILE_BYTES(                                                            \
      opname, shape_fn, [] { return (int64_t)0; }, state_arg)

// Like OP_PROFILE but also records the op's data footprint (bytes moved) so the
// [PERF] table can report achieved GB/s and % of the memory roofline. bytes_fn
// is a callable returning int64_t, invoked only when profiling is active.
#define OP_PROFILE_BYTES(opname, shape_fn, bytes_fn, state_arg)                \
  rgp_capture_fence(opname);                                                   \
  std::optional<OpProfileScope> _opProf;                                       \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    auto *_stream =                                                            \
        static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state_arg));       \
    if (_ps && _stream && op_profile_is_active(_ps)) {                         \
      op_profile_ensure_epoch(_ps, _stream);                                   \
      int _mkIdx = op_profile_acquire_marker(_ps);                             \
      _opProf.emplace(_ps, opname, (shape_fn)(), _stream, _mkIdx,              \
                      (int64_t)(bytes_fn)());                                  \
    }                                                                          \
  }

#define OP_PROFILE_CPU(opname, state_arg)                                      \
  rgp_capture_fence(opname);                                                   \
  std::optional<OpProfileCpuScope> _opProfCpu;                                 \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    if (_ps)                                                                   \
      _opProfCpu.emplace(_ps, opname);                                         \
  }
