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
                            const std::string &shape, int eventIndex,
                            double cpuMs, int64_t bytes = 0);
// Roofline peak (GB/s) used for the [PERF] %peak column. Reads
// HIPDNN_EP_ROOFLINE_GBPS once; defaults to 256 (Strix Halo LPDDR5X).
double op_profile_roofline_gbps();
void op_profile_add_cpu(OpProfileState *ps, const std::string &name,
                        double cpuMs);
bool op_profile_is_active(OpProfileState *ps);
int op_profile_acquire_event_pair(OpProfileState *ps);
hipEvent_t op_profile_get_start_event(OpProfileState *ps, int index);
hipEvent_t op_profile_get_stop_event(OpProfileState *ps, int index);
// One-time diagnostic for HIP-event failures on the profiling path.
void op_profile_note_event_error(const char *phase, int err);

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
  int eventIndex;
  hipStream_t stream;
  int64_t bytes;
  std::chrono::steady_clock::time_point cpuStart;

  OpProfileScope(OpProfileState *p, std::string n, std::string sh,
                 hipStream_t s, int evIdx, int64_t b = 0)
      : ps(p), name(std::move(n)), shape(std::move(sh)), eventIndex(evIdx),
        stream(s), bytes(b) {
    // Sync-isolated diagnostic mode: drain the stream BEFORE we start timing,
    // so this op's reported GPU time excludes any work queued ahead of us.
    // Pairs with the post-stop sync in the destructor to give standalone
    // per-op timings (concurrency is killed by design -- diagnostic only).
    if (hipdnn_ep_perf_isolate_enabled())
      hipStreamSynchronize(stream);
    cpuStart = std::chrono::steady_clock::now();
    hipError_t _e = hipEventRecord(op_profile_get_start_event(ps, eventIndex), stream);
    if (_e != hipSuccess)
      op_profile_note_event_error("record_start", _e);
    // On this ROCm build hipEventRecord can leave sticky error state even when
    // it returns hipSuccess; consume it so the next kernel wrapper that calls
    // hipGetLastError() (hip_reduce_sum / elementwise) doesn't misreport it as
    // its own "invalid resource handle" launch failure. The per-op timing is
    // unaffected (events still record correctly).
    (void)hipGetLastError();
  }

  ~OpProfileScope() {
    hipError_t _e = hipEventRecord(op_profile_get_stop_event(ps, eventIndex), stream);
    if (_e != hipSuccess)
      op_profile_note_event_error("record_stop", _e);
    (void)hipGetLastError(); // consume hipEventRecord sticky artifact (see ctor)
    if (hipdnn_ep_perf_isolate_enabled())
      hipStreamSynchronize(stream);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cpuStart)
                    .count();
    if (ps)
      op_profile_add_pending(ps, name, shape, eventIndex, ms, bytes);
  }

  OpProfileScope(const OpProfileScope &) = delete;
  OpProfileScope &operator=(const OpProfileScope &) = delete;
};

#define OP_PROFILE(opname, shape_fn, state_arg)                                \
  OP_PROFILE_BYTES(opname, shape_fn, [] { return (int64_t)0; }, state_arg)

// Like OP_PROFILE but also records the op's data footprint (bytes moved) so the
// [PERF] table can report achieved GB/s and % of the memory roofline. bytes_fn
// is a callable returning int64_t, invoked only when profiling is active.
#define OP_PROFILE_BYTES(opname, shape_fn, bytes_fn, state_arg)                 \
  std::optional<OpProfileScope> _opProf;                                       \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    auto *_stream =                                                            \
        static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state_arg));       \
    if (_ps && _stream && op_profile_is_active(_ps)) {                         \
      int _evIdx = op_profile_acquire_event_pair(_ps);                         \
      _opProf.emplace(_ps, opname, (shape_fn)(), _stream, _evIdx,              \
                      (int64_t)(bytes_fn)());                                   \
    }                                                                          \
  }

#define OP_PROFILE_CPU(opname, state_arg)                                      \
  std::optional<OpProfileCpuScope> _opProfCpu;                                 \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    if (_ps)                                                                   \
      _opProfCpu.emplace(_ps, opname);                                         \
  }
