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
// Per-iter iter update is stream-ordered, not host-store-driven: each iter
// enqueues an 8-byte hipMemcpyAsync(H2D) from a persistent pinned host
// staging array (cpu_buf[i] = i, filled once on grow) into a small device-
// side iter slot, then enqueues the body. Both ops on the same stream, so
// the body kernel reads the value placed by the matching memcpy. This
// replaces an earlier host-mapped + atomic_thread_fence(release) design
// which was unsafe: HIP does not order plain host stores against later
// stream submissions, so all M body kernels saw whatever value the host
// happened to leave in the mapped page (typically M-1, after the host
// finished its loop before any kernel ran). The new design adds a single
// hipMemcpyAsync enqueue (~1-2 µs) per iter -- still vastly cheaper than
// hipStreamSynchronize (~50-200 µs) and preserves the counted-loop fast
// path's pipelined throughput.
//
// cond is still host-mapped: cond_init is host-written once before the
// loop starts, cond_out is GPU-written per iter, and the dynamic path
// uses a reusable hipEvent_t (hipEventDisableTiming) recorded on the
// stream and hipEventSynchronize'd before reading cond_host. The event
// sync serialises kernel completion vs the host read, so the read sees
// the cond_out written by the just-finished iter.
//
// Aliasing invariant: each v_in_i and v_out_i (and, on the dynamic path,
// cond_in and cond_out) refer to the same memref slot in the body's call.
// Safe under the v1 body semantics where each kernel touches each cell at
// most once per launch. Single-level nesting is enforced at compile time
// by OnnxLoopOutlinePass (rejects nested onnx.Loop), so the shared
// per-state iter/cond buffers cannot race here.
//
//===----------------------------------------------------------------------===//

#include "debug_log.h"
#include "hipdnn_ep_errors.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// Lazily allocate the per-state iter / cond buffers + reusable sync event
// on first loop call, then grow the iter cpu staging array on demand if
// max_trip_count exceeds its current capacity.
//
// iter side (stream-ordered, see runtime_state_internal.h comment):
//   * loop_iter_cpu_buf : pinned host int64[capacity], values [i] = i,
//                         filled once at grow time
//   * loop_iter_dev     : device int64, target of per-iter hipMemcpyAsync
//
// cond side stays host-mapped (correct as-is; only writer is host-before-loop
// or GPU-via-stream, both serialised).
//
// Partial-allocation policy: if any later step fails, earlier successful
// allocations stay live on the RuntimeState so the next call's null-check
// skips re-allocating them -- they are correctly freed in
// hipdnn_ep_state_cleanup regardless. This is intentional; do not "clean
// up" by nullifying earlier slots on a later failure, or repeated transient
// failures will leak.
int ensureLoopBuffers(RuntimeState *state, int64_t max_trip_count) {
  // Grow-on-demand the pinned host iter array. Static contents: cpu_buf[i] = i.
  if (max_trip_count > 0 &&
      static_cast<size_t>(max_trip_count) > state->loop_iter_capacity) {
    size_t old_cap = state->loop_iter_capacity;
    size_t new_cap = static_cast<size_t>(max_trip_count);
    void *new_buf = nullptr;
    if (hipHostMalloc(&new_buf, new_cap * sizeof(int64_t),
                      hipHostMallocDefault) != hipSuccess) {
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipHostMalloc for iter cpu buf (%zu "
          "entries) failed\n",
          new_cap);
      return -1;
    }
    int64_t *new_arr = static_cast<int64_t *>(new_buf);
    // Copy preserved [0..old_cap) -- values are static (i -> i), so even if we
    // re-init from scratch the result is identical; copy is just a hair faster
    // and removes any future surprise if cpu_buf semantics evolve.
    if (state->loop_iter_cpu_buf && old_cap > 0) {
      std::memcpy(new_arr, state->loop_iter_cpu_buf, old_cap * sizeof(int64_t));
    }
    for (size_t i = old_cap; i < new_cap; ++i) {
      new_arr[i] = static_cast<int64_t>(i);
    }
    if (state->loop_iter_cpu_buf) {
      // Stream may still hold async memcpy reads from the old buf; drain
      // before freeing. Grow is a rare per-session event so the sync cost
      // is negligible.
      hipStreamSynchronize(static_cast<hipStream_t>(state->stream));
      hipHostFree(state->loop_iter_cpu_buf);
    }
    state->loop_iter_cpu_buf = new_buf;
    state->loop_iter_capacity = new_cap;
  }
  if (!state->loop_iter_dev) {
    if (hipMalloc(&state->loop_iter_dev, sizeof(int64_t)) != hipSuccess) {
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipMalloc for iter dev buffer failed\n");
      state->loop_iter_dev = nullptr;
      return -1;
    }
  }
  if (!state->loop_cond_host) {
    // Bool stored as 1 byte (matches LLVM bool ABI and memref<i1> layout).
    if (hipHostMalloc(&state->loop_cond_host, sizeof(int8_t),
                      hipHostMallocMapped) != hipSuccess) {
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipHostMalloc for cond buffer failed\n");
      state->loop_cond_host = nullptr;
      return -1;
    }
    if (hipHostGetDevicePointer(&state->loop_cond_dev, state->loop_cond_host,
                                0) != hipSuccess) {
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipHostGetDevicePointer for cond "
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
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipEventCreateWithFlags failed\n");
      return -1;
    }
    state->loop_event = static_cast<void *>(evt);
  }
  return 0;
}

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
  // ONNX Loop: when cond_init is false the body must execute zero times,
  // regardless of M. On the dynamic path this falls out of the per-iter
  // cond check, but on the counted path consultsCond() is false and the
  // body would otherwise execute M times -- the outlining pass detected
  // cond_out == cond_in at SSA level, which doesn't constrain the runtime
  // cond_init value. Short-circuit here so both paths honor the spec.
  if (!cond_init)
    return 0;
  if (ensureLoopBuffers(state, max_trip_count) != 0)
    return -1;

  void *iter_dev = state->loop_iter_dev;
  void *cond_dev = state->loop_cond_dev;
  int64_t *iter_cpu_buf = static_cast<int64_t *>(state->loop_iter_cpu_buf);
  int8_t *cond_host = static_cast<int8_t *>(state->loop_cond_host);
  hipStream_t stream = static_cast<hipStream_t>(state->stream);

  // Initialize cond_in for the body to read. cond_init is guaranteed true
  // here (false case short-circuited above). Host writes cond_host once,
  // before any kernel launches read cond_dev, so no host-vs-stream race
  // (the launch of iter 0 below already drains any prior writer).
  *cond_host = int8_t{1};

  bool keep_going = true;
  for (int64_t i = 0; i < max_trip_count; ++i) {
    if constexpr (CondPolicy::consultsCond()) {
      if (!keep_going)
        break;
    }
    // Stream-order the iter update: enqueue an 8-byte H2D copy from the
    // persistent pinned `cpu_buf[i]` (statically holds value `i`) into the
    // body's iter_dev slot. The subsequent body_fn enqueues its kernels on
    // the same stream, so they observe `*iter_dev == i` (the value placed
    // by this matching memcpy). No per-iter CPU sync, no host-store-vs-
    // kernel-launch race that the old `*iter_host = i; fence;` design had.
    if (hipMemcpyAsync(iter_dev, &iter_cpu_buf[i], sizeof(int64_t),
                       hipMemcpyHostToDevice, stream) != hipSuccess) {
      hipdnn_ep_log_emit(
          "hipdnn_ep_run_*_loop: hipMemcpyAsync for iter[%lld] failed\n",
          static_cast<long long>(i));
      return -1;
    }
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
