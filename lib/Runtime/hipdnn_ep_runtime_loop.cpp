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
//                                  Skips per-iter cond readback; the loop
//                                  reduces to `for (i = 0; i < M; ++i)`.
//   * hipdnn_ep_run_loop         : slow path. Mirrors ORT CUDA EP +
//                                  MIGraphX behavior: reads the body's
//                                  cond_out every iter to decide whether
//                                  to continue.
//
// Both drivers reuse a pair of host-mapped per-state buffers (int64 iter,
// int8 cond) so the host can write iter and read cond with zero DMA.
// Stream-ordering between the host store of iter and the body kernels
// keeps the GPU view consistent without an explicit sync. For the
// dynamic path's cond readback, a reusable hipEvent_t (created with
// hipEventDisableTiming) is recorded on the stream after each body call
// and synchronized on before the host reads cond -- lower latency than
// hipStreamSynchronize for the small-body case that dominates ONNX-Loop
// usage, and the same readback shape as the seqlens_k optimization in
// gqa.cpp.
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

// Lazily allocate the per-state iter / cond host-mapped buffers + reusable
// sync event on first loop call. Both buffers are tiny (8 + 1 bytes); the
// event is allocator-managed and reused across iters. All three are freed
// once in hipdnn_ep_state_cleanup.
//
// Partial-allocation policy: if any later step fails, earlier successful
// allocations stay live on the RuntimeState so the next call's null-check
// skips re-allocating them -- they are correctly freed in
// hipdnn_ep_state_cleanup regardless. This is intentional; do not "clean
// up" by nullifying earlier slots on a later failure, or repeated transient
// failures will leak.
int ensureLoopBuffers(RuntimeState *state) {
  if (!state->loop_iter_host) {
    if (hipHostMalloc(&state->loop_iter_host, sizeof(int64_t),
                      hipHostMallocMapped) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: hipHostMalloc for iter buffer failed\n");
      state->loop_iter_host = nullptr;
      return -1;
    }
    if (hipHostGetDevicePointer(&state->loop_iter_dev, state->loop_iter_host,
                                0) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_run_*_loop: hipHostGetDevicePointer for iter "
                      "buffer failed\n");
      hipHostFree(state->loop_iter_host);
      state->loop_iter_host = nullptr;
      state->loop_iter_dev = nullptr;
      return -1;
    }
  }
  if (!state->loop_cond_host) {
    // Bool stored as 1 byte (matches LLVM bool ABI and memref<i1> layout).
    if (hipHostMalloc(&state->loop_cond_host, sizeof(int8_t),
                      hipHostMallocMapped) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: hipHostMalloc for cond buffer failed\n");
      state->loop_cond_host = nullptr;
      return -1;
    }
    if (hipHostGetDevicePointer(&state->loop_cond_dev, state->loop_cond_host,
                                0) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_run_*_loop: hipHostGetDevicePointer for cond "
                      "buffer failed\n");
      hipHostFree(state->loop_cond_host);
      state->loop_cond_host = nullptr;
      state->loop_cond_dev = nullptr;
      return -1;
    }
  }
  if (!state->loop_event) {
    hipEvent_t evt = nullptr;
    if (hipEventCreateWithFlags(&evt, hipEventDisableTiming) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_run_*_loop: hipEventCreateWithFlags failed\n");
      return -1;
    }
    state->loop_event = static_cast<void *>(evt);
  }
  return 0;
}

// RAII guard that enforces the single-level loop nesting invariant. The
// driver shares one iter/cond buffer pair per RuntimeState, so a loop
// body that itself launches a hip.loop would silently overwrite the
// outer driver's buffers before the outer reads them on its next iter.
// Hard-fail at entry rather than corrupt silently. v1 limit; a future
// extension can replace the shared buffers with a per-depth stack.
struct LoopNestingGuard {
  RuntimeState *state;
  bool ok;
  explicit LoopNestingGuard(RuntimeState *s) : state(s), ok(false) {
    if (state->loop_nesting_depth >= 1) {
      fprintf(stderr, "hipdnn_ep_run_*_loop: nested loops not supported in v1 "
                      "(loop iter/cond buffers are shared across the "
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
// reads cond from the device. The dynamic policy records an event on the
// stream after the body and synchronizes on it before reading cond_host.
//
// consultsCond() is constexpr so `if constexpr` at the call sites elides
// the entire cond-handling branch in the counted-path instantiation --
// including the indirect call to checkCond -- without relying on cross-TU
// LTO to inline through the runtime DLL boundary.
struct CountedCondPolicy {
  static int checkCond(RuntimeState * /*state*/, int8_t * /*cond_host*/,
                       bool * /*out_continue*/) {
    return 0;
  }
  static constexpr bool consultsCond() { return false; }
};

struct DynamicCondPolicy {
  // Post-body: record + sync on a reusable event, then read the host-mapped
  // cond byte directly. This is ONNX-Loop slow-path cost; ORT CUDA EP
  // `LoopImpl::Execute` and MIGraphX `run_loop` are equivalent (and pre-
  // host-mapped MIGraphX also paid hipMemcpyAsync(D2H) + hipStreamSync per
  // iter on top of this).
  static int checkCond(RuntimeState *state, int8_t *cond_host,
                       bool *out_continue) {
    hipEvent_t evt = static_cast<hipEvent_t>(state->loop_event);
    hipStream_t stream = static_cast<hipStream_t>(state->stream);
    if (hipEventRecord(evt, stream) != hipSuccess)
      return -1;
    if (hipEventSynchronize(evt) != hipSuccess)
      return -1;
    *out_continue = (*cond_host != 0);
    return 0;
  }
  static constexpr bool consultsCond() { return true; }
};

// Backend-agnostic templated driver. The policy resolves the cond-handling
// strategy at compile time so the counted path emits zero per-iter sync
// and has no per-iter branch on the policy.
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
  if (ensureLoopBuffers(state) != 0)
    return -1;

  void *iter_dev = state->loop_iter_dev;
  void *cond_dev = state->loop_cond_dev;
  int64_t *iter_host = static_cast<int64_t *>(state->loop_iter_host);
  int8_t *cond_host = static_cast<int8_t *>(state->loop_cond_host);

  // Initialize cond with a direct host store -- the kernel's cond_in arg
  // may be read by the body even on the counted path (no per-iter readback,
  // but a single read at body entry is still legal SSA), so initialize
  // unconditionally. No memcpy: the GPU sees the value via the mapped
  // pointer once any subsequent kernel launches on the same stream.
  *cond_host = cond_init ? int8_t{1} : int8_t{0};

  bool keep_going = cond_init;
  for (int64_t i = 0; i < max_trip_count; ++i) {
    if constexpr (CondPolicy::consultsCond()) {
      if (!keep_going)
        break;
    }
    // Direct host store; the GPU reads via iter_dev (the device-mapped
    // alias of loop_iter_host). HIP stream ordering guarantees the host
    // store is visible to any kernel launched on `state->stream` after
    // this point -- no hipMemcpyAsync(H2D, 8B) per iter, and no host
    // -stack-source race that the previous `&i`-based memcpy exposed
    // (small-size copies are usually staged synchronously by the driver
    // in practice, but the spec allows the read to be deferred).
    *iter_host = i;
    int rc =
        body_fn(state, iter_dev, cond_dev, loop_carried_descs, capture_descs);
    if (rc != 0)
      return rc;
    if constexpr (CondPolicy::consultsCond()) {
      if (CondPolicy::checkCond(state, cond_host, &keep_going) != 0)
        return -1;
    }
  }
  return 0;
}

} // namespace

extern "C" int
hipdnn_ep_run_counted_loop(RuntimeState *state, HipdnnEpLoopBodyFn body_fn,
                           int64_t max_trip_count, bool cond_init,
                           int32_t num_loop_carried, int32_t num_captures,
                           void **loop_carried_descs, void **capture_descs) {
  return runLoopImpl<CountedCondPolicy>(
      state, body_fn, max_trip_count, cond_init, num_loop_carried, num_captures,
      loop_carried_descs, capture_descs);
}

extern "C" int
hipdnn_ep_run_loop(RuntimeState *state, HipdnnEpLoopBodyFn body_fn,
                   int64_t max_trip_count, bool cond_init,
                   int32_t num_loop_carried, int32_t num_captures,
                   void **loop_carried_descs, void **capture_descs) {
  return runLoopImpl<DynamicCondPolicy>(
      state, body_fn, max_trip_count, cond_init, num_loop_carried, num_captures,
      loop_carried_descs, capture_descs);
}
