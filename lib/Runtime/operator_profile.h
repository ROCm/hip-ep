/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- operator_profile.h - Per-operator GPU timing instrumentation ------===//
//
// Lightweight HIP-event based profiler that reports per-operator wall-clock
// time on the runtime stream.
//
// Enable by setting environment variable HIPDNN_EP_OP_PROFILE=1 before
// loading the EP. When disabled, the OpScope constructor/destructor are
// effectively no-ops (a single cached-bool branch in the dynamic library).
//
// Per inference, the profiler records a (start, end) hipEvent pair around
// each wrap_<op>(...) call, accumulates them in a pending list, and on the
// next inference boundary (or session cleanup):
//   1. Synchronizes on the last recorded event so all prior events are valid
//   2. Reads elapsed times via hipEventElapsedTime
//   3. Routes the sample into the per-session WARMUP or STEADY-STATE bucket
//      based on inference index (controlled by HIPDNN_EP_PERF_WARMUP, the
//      same env var used by [PERF] -- the two reports partition identically)
//   4. Returns the events to a recyclable pool
//
// At session_cleanup the WARMUP and STEADY-STATE buckets are emitted as
// separate "[OP_PROFILE] session #N WARMUP" / "STEADY-STATE" tables. Each
// table mixes compute operators and I/O (memcpy) entries sorted together
// by total_ms descending, with I/O-specific columns (bytes, GB/s) blank
// for compute rows. This lets the user spot hot rows regardless of kind
// and see what fraction of inference time is I/O vs compute at a glance.
// Splitting warmup off keeps hipBLASLt heuristic search, MIOpen
// kernel-finder cache fills, hipRTC JIT, and the first hipMalloc growth
// from inflating the steady-state averages.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_OPERATOR_PROFILE_H
#define HIPDNN_EP_OPERATOR_PROFILE_H

#include <cstddef>

struct RuntimeState;

namespace hipdnn_ep {

// True when HIPDNN_EP_OP_PROFILE env var is set to a non-zero value at
// process startup. Cached on first call.
bool op_profile_enabled();

// Begin a timed scope on the runtime's GPU stream. Records a hipEvent.
// `name` must point to a C string with static storage duration (typically
// __func__). Returns an opaque token to pair with op_profile_scope_end.
// Returns nullptr when profiling is disabled or the event could not be
// recorded -- callers must always pair with op_profile_scope_end regardless.
void *op_profile_scope_begin(RuntimeState *state, const char *name);

// End the timed scope opened by op_profile_scope_begin.
void op_profile_scope_end(RuntimeState *state, const char *name, void *token);

// I/O variant: like op_profile_scope_begin/end, but also records the byte
// count moved by the operation. In the unified report these entries
// share the same table as compute operators, with extra bytes and GB/s
// columns populated (GB/s = total_bytes / total_ms / 1e6). Sorting is by
// total_ms so memcpy hotspots float to the top right alongside kernel
// hotspots.
//
// Use for wrap_<op>() functions whose primary work is a memory transfer
// (H2D/D2H/D2D), and for the framework's per-input/per-output transfers.
void *op_profile_io_scope_begin(RuntimeState *state, const char *name,
                                size_t bytes);
void op_profile_io_scope_end(RuntimeState *state, const char *name, void *token,
                             size_t bytes);

// Sync pending events, aggregate them into per-operator stats for the
// current inference, print a summary, and recycle the events. Safe to call
// when nothing is pending.
//
// The runtime invokes this at the start of each new inference (so the
// previous inference's events have completed when their elapsed time is
// queried) and again from state_cleanup.
void op_profile_flush_inference();

// Print the lifetime summary across all inferences and reset the
// accumulator. Frees pooled hipEvents. Called from state_cleanup.
void op_profile_print_summary_and_reset();

// RAII guard inserted at the top of each wrap_<op>() function.
class OpScope {
public:
  OpScope(RuntimeState *state, const char *name)
      : state_(state), name_(name),
        token_(op_profile_scope_begin(state, name)) {}
  ~OpScope() { op_profile_scope_end(state_, name_, token_); }
  OpScope(const OpScope &) = delete;
  OpScope &operator=(const OpScope &) = delete;

private:
  RuntimeState *state_;
  const char *name_;
  void *token_;
};

// RAII guard for memory-copy operations. Records bytes alongside time so
// the reporter can compute bandwidth.
class IoScope {
public:
  IoScope(RuntimeState *state, const char *name, size_t bytes)
      : state_(state), name_(name), bytes_(bytes),
        token_(op_profile_io_scope_begin(state, name, bytes)) {}
  ~IoScope() { op_profile_io_scope_end(state_, name_, token_, bytes_); }
  IoScope(const IoScope &) = delete;
  IoScope &operator=(const IoScope &) = delete;

private:
  RuntimeState *state_;
  const char *name_;
  size_t bytes_;
  void *token_;
};

} // namespace hipdnn_ep

// Convenience macro: declares an OpScope using __func__ as the operator name.
// Use at the top of every wrap_<op>(RuntimeState *state, ...) function body.
#define HIPDNN_EP_OP_PROFILE_SCOPE(state)                                      \
  hipdnn_ep::OpScope _hipdnn_ep_op_scope((state), __func__)

// Convenience macro for memory-copy sites. `name` must be a static C string
// (e.g. a string literal) and `bytes` is the size of the transfer in bytes.
// Records both elapsed time and bytes moved for bandwidth reporting.
#define HIPDNN_EP_IO_PROFILE_SCOPE(state, name, bytes)                         \
  hipdnn_ep::IoScope _hipdnn_ep_io_scope((state), (name), (bytes))

#endif // HIPDNN_EP_OPERATOR_PROFILE_H
