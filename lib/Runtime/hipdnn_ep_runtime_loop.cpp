/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- hipdnn_ep_runtime_loop.cpp - ONNX Loop drivers -------------------===//
//
// Runtime drivers for the `hip.loop` lowering -- one CPU-driven iteration
// loop per call.  Mirrors MIGraphX's `run_loop.hpp` policy-based design
// in spirit: a single backend-agnostic templated driver, specialized at
// compile time via a CondPolicy.  Two specializations are exported:
//
//   * hipdnn_ep_run_counted_loop : fast path. Used by the HipToLLVM
//                                  lowering when the outlining pass proves
//                                  cond_out == cond_in (SSA-equality).
//                                  Skips per-iter D2H cond sync; the loop
//                                  reduces to `for (i = 0; i < M; ++i)`.
//   * hipdnn_ep_run_loop         : slow path. Mirrors ORT CUDA EP +
//                                  MIGraphX behavior: D2H-syncs cond_out
//                                  every iter to decide whether to
//                                  continue.
//
// Both drivers reuse a pair of tiny per-state device buffers (one int64
// for iter, one int8 for cond) so we don't hipMalloc/hipFree on every
// inference. Lazily allocated; freed in hipdnn_ep_state_cleanup.
//
// See `.cursor/plans/onnx-loop-support.plan.md` (P4/P5) for the trampoline
// ABI rationale and the aliasing invariant that allows v_in and v_out
// (resp. cond_in and cond_out) to share the same buffer.
//
//===----------------------------------------------------------------------===//

#include "debug_log.h"
#include "hipdnn_ep_errors.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

namespace {

// Lazily allocate the per-state iter / cond device buffers on first loop
// call. Both are tiny (8 + 1 bytes) so we don't bother with grow-on-demand
// -- they're freed once in hipdnn_ep_state_cleanup.
//
// Partial-allocation policy: if iter succeeds but cond fails, we leave
// `state->loop_iter_dev_buf` non-null so the next call's null-check skips
// re-allocating it -- the iter buffer is correctly freed in
// `hipdnn_ep_state_cleanup` regardless. This is intentional; do not "clean
// up" by nullifying iter on cond failure, or repeated transient failures
// will leak.
int ensureLoopDeviceBuffers(RuntimeState *state) {
  if (!state->loop_iter_dev_buf) {
    if (hipMalloc(&state->loop_iter_dev_buf, sizeof(int64_t)) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: hipMalloc for iter buffer failed\n");
      state->loop_iter_dev_buf = nullptr;
      return -1;
    }
  }
  if (!state->loop_cond_dev_buf) {
    // Bool stored as 1 byte (matches LLVM bool ABI and memref<i1> layout).
    if (hipMalloc(&state->loop_cond_dev_buf, sizeof(int8_t)) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: hipMalloc for cond buffer failed\n");
      state->loop_cond_dev_buf = nullptr;
      return -1;
    }
  }
  return 0;
}

// RAII guard that enforces the single-level loop nesting invariant. The
// driver shares one iter/cond device buffer pair per RuntimeState, so a
// loop body that itself launches a hip.loop would silently overwrite the
// outer driver's buffers before the outer reads them on its next iter.
// Hard-fail at entry rather than corrupt silently. v1 limit; a future
// extension can replace the shared buffers with a per-depth stack.
struct LoopNestingGuard {
  RuntimeState *state;
  bool ok;
  explicit LoopNestingGuard(RuntimeState *s) : state(s), ok(false) {
    if (state->loop_nesting_depth >= 1) {
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: nested loops not supported in v1 "
              "(loop_iter/cond device buffers are shared across the "
              "RuntimeState)\n");
      return;
    }
    state->loop_nesting_depth++;
    ok = true;
  }
  ~LoopNestingGuard() {
    if (ok)
      state->loop_nesting_depth--;
  }
};

// Policy: each iter, decide whether to continue. The counted policy never
// reads cond from the device (trivial true while iter < M). The dynamic
// policy issues a D2H + stream-sync to read the body's cond_out.
struct CountedCondPolicy {
  // Pre-loop: just write the initial cond once.
  static int initCond(RuntimeState *state, bool cond_init, void *cond_dev) {
    int8_t cond_val = cond_init ? int8_t{1} : int8_t{0};
    if (hipMemcpyAsync(cond_dev, &cond_val, sizeof(int8_t),
                        hipMemcpyHostToDevice,
                        static_cast<hipStream_t>(state->stream)) != hipSuccess)
      return -1;
    return 0;
  }
  // Post-body: counted loop never reads cond.
  static int checkCond(RuntimeState * /*state*/, void * /*cond_dev*/,
                       bool * /*out_continue*/) {
    return 0;
  }
  static bool consultsCond() { return false; }
};

struct DynamicCondPolicy {
  static int initCond(RuntimeState *state, bool cond_init, void *cond_dev) {
    int8_t cond_val = cond_init ? int8_t{1} : int8_t{0};
    if (hipMemcpyAsync(cond_dev, &cond_val, sizeof(int8_t),
                        hipMemcpyHostToDevice,
                        static_cast<hipStream_t>(state->stream)) != hipSuccess)
      return -1;
    return 0;
  }
  // Post-body: D2H the body's cond_out and stream-sync so the host bool
  // is visible. This is the cost ONNX-Loop slow path always pays; ORT
  // CUDA EP `LoopImpl::Execute` and MIGraphX `run_loop` are equivalent.
  static int checkCond(RuntimeState *state, void *cond_dev,
                       bool *out_continue) {
    int8_t cond_val = 0;
    if (hipMemcpyAsync(&cond_val, cond_dev, sizeof(int8_t),
                        hipMemcpyDeviceToHost,
                        static_cast<hipStream_t>(state->stream)) != hipSuccess)
      return -1;
    if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
        hipSuccess)
      return -1;
    *out_continue = (cond_val != 0);
    return 0;
  }
  static bool consultsCond() { return true; }
};

// Backend-agnostic templated driver. The policy resolves the cond-handling
// strategy at compile time so the counted path emits zero D2H syncs and
// has no per-iter branch on the policy.
template <class CondPolicy>
int runLoopImpl(RuntimeState *state, HipdnnEpLoopBodyFn body_fn,
                int64_t max_trip_count, bool cond_init,
                int32_t /*num_loop_carried*/, int32_t /*num_captures*/,
                void **loop_carried_descs, void **capture_descs) {
  if (!state || !body_fn)
    return -1;
  if (max_trip_count < 0)
    return -1;
  LoopNestingGuard nestGuard(state);
  if (!nestGuard.ok)
    return -1;
  if (ensureLoopDeviceBuffers(state) != 0)
    return -1;

  void *iter_dev = state->loop_iter_dev_buf;
  void *cond_dev = state->loop_cond_dev_buf;
  hipStream_t stream = static_cast<hipStream_t>(state->stream);

  if (CondPolicy::initCond(state, cond_init, cond_dev) != 0)
    return -1;

  bool keep_going = cond_init;
  for (int64_t i = 0; i < max_trip_count; ++i) {
    if (CondPolicy::consultsCond() && !keep_going)
      break;
    // Write the host iter value to the device iter buffer.
    if (hipMemcpyAsync(iter_dev, &i, sizeof(int64_t), hipMemcpyHostToDevice,
                        stream) != hipSuccess)
      return -1;
    int rc = body_fn(state, iter_dev, cond_dev, loop_carried_descs,
                     capture_descs);
    if (rc != 0)
      return rc;
    if (CondPolicy::checkCond(state, cond_dev, &keep_going) != 0)
      return -1;
  }
  return 0;
}

} // namespace

extern "C" int hipdnn_ep_run_counted_loop(
    RuntimeState *state, HipdnnEpLoopBodyFn body_fn, int64_t max_trip_count,
    bool cond_init, int32_t num_loop_carried, int32_t num_captures,
    void **loop_carried_descs, void **capture_descs) {
  return runLoopImpl<CountedCondPolicy>(state, body_fn, max_trip_count,
                                          cond_init, num_loop_carried,
                                          num_captures, loop_carried_descs,
                                          capture_descs);
}

extern "C" int hipdnn_ep_run_loop(RuntimeState *state,
                                   HipdnnEpLoopBodyFn body_fn,
                                   int64_t max_trip_count, bool cond_init,
                                   int32_t num_loop_carried,
                                   int32_t num_captures,
                                   void **loop_carried_descs,
                                   void **capture_descs) {
  return runLoopImpl<DynamicCondPolicy>(state, body_fn, max_trip_count,
                                          cond_init, num_loop_carried,
                                          num_captures, loop_carried_descs,
                                          capture_descs);
}
