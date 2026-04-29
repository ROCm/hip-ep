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
                            const std::string &shape, hipEvent_t start,
                            hipEvent_t stop, double cpuMs);
void op_profile_add_cpu(OpProfileState *ps, const std::string &name,
                        double cpuMs);
bool op_profile_is_active(OpProfileState *ps);

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
  hipEvent_t start, stop;
  hipStream_t stream;
  std::chrono::steady_clock::time_point cpuStart;

  OpProfileScope(OpProfileState *p, std::string n, std::string sh,
                 hipStream_t s)
      : ps(p), name(std::move(n)), shape(std::move(sh)), stream(s),
        cpuStart(std::chrono::steady_clock::now()) {
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, stream);
  }

  ~OpProfileScope() {
    hipEventRecord(stop, stream);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - cpuStart)
                    .count();
    if (ps)
      op_profile_add_pending(ps, name, shape, start, stop, ms);
  }

  OpProfileScope(const OpProfileScope &) = delete;
  OpProfileScope &operator=(const OpProfileScope &) = delete;
};

#define OP_PROFILE(opname, shape_fn, state_arg)                                \
  std::optional<OpProfileScope> _opProf;                                       \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    auto *_stream =                                                            \
        static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state_arg));       \
    if (_ps && _stream && op_profile_is_active(_ps))                           \
      _opProf.emplace(_ps, opname, (shape_fn)(), _stream);                     \
  }

#define OP_PROFILE_CPU(opname, state_arg)                                      \
  std::optional<OpProfileCpuScope> _opProfCpu;                                 \
  if (hipdnn_ep_perf_enabled()) {                                              \
    auto *_ps = static_cast<OpProfileState *>(                                 \
        hipdnn_ep_state_get_op_profile(state_arg));                            \
    if (_ps)                                                                   \
      _opProfCpu.emplace(_ps, opname);                                         \
  }
