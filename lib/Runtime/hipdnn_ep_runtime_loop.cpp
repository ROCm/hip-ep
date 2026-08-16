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
// Every invocation owns a frame with independent iter/condition storage,
// synchronization event, and two high-water banks per carrier. The callback
// reads one complete current descriptor set and writes a separate next set;
// the driver swaps only after success. This makes dynamic extents, nesting,
// repeated calls, and simultaneous driver invocations independent.
//
//===----------------------------------------------------------------------===//

#include "debug_log.h"
#include "hipdnn_ep_errors.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

struct HipdnnEpLoopFrame {
  struct CarrierBanks {
    void *data[2];
    size_t capacity[2];
    const void *current_data;
    int current_bank;
    int allocated_bank;
    int published_bank;
  };

  RuntimeState *state;
  int32_t num_carriers;
  CarrierBanks *carriers;
  void *iter_cpu_buf;
  void *iter_dev;
  void *cond_host;
  void *cond_dev;
  void *event;
  int status;
  HipdnnEpLoopFrame *parent;
  HipdnnEpLoopFrame *first_child;
  HipdnnEpLoopFrame *next_sibling;
  HipdnnEpLoopFrame *quarantine_next;
};

namespace {

#ifdef HIPDNN_EP_RUNTIME_TESTING
bool failNextLoopSync = false;
bool failNextLoopAlloc = false;
#endif

hipError_t synchronizeLoopStream(hipStream_t stream) {
#ifdef HIPDNN_EP_RUNTIME_TESTING
  if (failNextLoopSync) {
    failNextLoopSync = false;
    return -1;
  }
#endif
  return hipStreamSynchronize(stream);
}

std::mutex *getFrameMutex(RuntimeState *state) {
  return static_cast<std::mutex *>(state ? state->loop_frames_mutex : nullptr);
}

void destroyFrame(HipdnnEpLoopFrame *frame) {
  if (!frame)
    return;
  if (frame->event)
    hipEventDestroy(static_cast<hipEvent_t>(frame->event));
  if (frame->iter_cpu_buf)
    hipHostFree(frame->iter_cpu_buf);
  if (frame->iter_dev)
    hipFree(frame->iter_dev);
  if (frame->cond_host)
    hipHostFree(frame->cond_host);
  if (frame->carriers) {
    for (int32_t i = 0; i < frame->num_carriers; ++i)
      for (int bank = 0; bank < 2; ++bank)
        if (frame->carriers[i].data[bank])
          hipFree(frame->carriers[i].data[bank]);
    std::free(frame->carriers);
  }
  while (frame->first_child) {
    HipdnnEpLoopFrame *child = frame->first_child;
    frame->first_child = child->next_sibling;
    child->parent = nullptr;
    destroyFrame(child);
  }
  std::free(frame);
}

HipdnnEpLoopFrame *createFrame(RuntimeState *state, int32_t num_carriers,
                               int64_t max_trip_count) {
  auto *frame = static_cast<HipdnnEpLoopFrame *>(
      std::calloc(1, sizeof(HipdnnEpLoopFrame)));
  if (!frame)
    return nullptr;
  frame->state = state;
  frame->num_carriers = num_carriers;
  if (num_carriers > 0) {
    frame->carriers = static_cast<HipdnnEpLoopFrame::CarrierBanks *>(
        std::calloc(static_cast<size_t>(num_carriers),
                    sizeof(HipdnnEpLoopFrame::CarrierBanks)));
    if (!frame->carriers) {
      destroyFrame(frame);
      return nullptr;
    }
    for (int32_t i = 0; i < num_carriers; ++i) {
      frame->carriers[i].current_bank = -1;
      frame->carriers[i].allocated_bank = -1;
      frame->carriers[i].published_bank = -2;
    }
  }
  if (max_trip_count > 0) {
    size_t count = static_cast<size_t>(max_trip_count);
    if (count > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
        hipHostMalloc(&frame->iter_cpu_buf, count * sizeof(int64_t),
                      hipHostMallocDefault) != hipSuccess) {
      destroyFrame(frame);
      return nullptr;
    }
    auto *iters = static_cast<int64_t *>(frame->iter_cpu_buf);
    for (int64_t i = 0; i < max_trip_count; ++i)
      iters[i] = i;
  }
  if (hipMalloc(&frame->iter_dev, sizeof(int64_t)) != hipSuccess ||
      hipHostMalloc(&frame->cond_host, sizeof(int8_t), hipHostMallocMapped) !=
          hipSuccess ||
      hipHostGetDevicePointer(&frame->cond_dev, frame->cond_host, 0) !=
          hipSuccess) {
    destroyFrame(frame);
    return nullptr;
  }
  hipEvent_t event = nullptr;
  if (hipEventCreateWithFlags(&event, hipEventDisableTiming) != hipSuccess) {
    destroyFrame(frame);
    return nullptr;
  }
  frame->event = static_cast<void *>(event);
  return frame;
}

void quarantineFrame(RuntimeState *state, HipdnnEpLoopFrame *frame) {
  std::mutex *mutex = getFrameMutex(state);
  if (!mutex) {
    destroyFrame(frame);
    return;
  }
  std::lock_guard<std::mutex> lock(*mutex);
  frame->quarantine_next =
      static_cast<HipdnnEpLoopFrame *>(state->quarantined_loop_frames);
  state->quarantined_loop_frames = frame;
}

void unlinkFromParent(HipdnnEpLoopFrame *frame) {
  if (!frame || !frame->parent)
    return;
  HipdnnEpLoopFrame **link = &frame->parent->first_child;
  while (*link && *link != frame)
    link = &(*link)->next_sibling;
  if (*link == frame)
    *link = frame->next_sibling;
  frame->parent = nullptr;
  frame->next_sibling = nullptr;
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
  static int checkCond(HipdnnEpLoopFrame * /*frame*/, int8_t * /*cond_host*/,
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
  static int checkCond(HipdnnEpLoopFrame *frame, int8_t *cond_host,
                       bool *out_continue) {
    hipEvent_t evt = static_cast<hipEvent_t>(frame->event);
    hipStream_t stream = static_cast<hipStream_t>(frame->state->stream);
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
                int32_t num_loop_carried, int32_t /*num_captures*/,
                void **initial_descs, void **scratch_descs_a,
                void **scratch_descs_b, void **capture_descs,
                HipdnnEpLoopFrame *parent_frame, void ***final_descs,
                HipdnnEpLoopFrame **frame_out) {
  if (!state || !body_fn || !final_descs || !frame_out)
    return -1;
  if (max_trip_count < 0 || num_loop_carried < 0 ||
      (num_loop_carried > 0 &&
       (!initial_descs || !scratch_descs_a || !scratch_descs_b)))
    return -1;
  *final_descs = initial_descs;
  *frame_out = nullptr;
  // ONNX Loop: when cond_init is false the body must execute zero times,
  // regardless of M. On the dynamic path this falls out of the per-iter
  // cond check, but on the counted path consultsCond() is false and the
  // body would otherwise execute M times -- the outlining pass detected
  // cond_out == cond_in at SSA level, which doesn't constrain the runtime
  // cond_init value. Short-circuit here so both paths honor the spec.
  if (!cond_init)
    return 0;
  if (max_trip_count == 0)
    return 0;

  HipdnnEpLoopFrame *frame =
      createFrame(state, num_loop_carried, max_trip_count);
  if (!frame) {
    (void)hipdnn_ep_state_set_error_flag(state);
    return -1;
  }

  void *iter_dev = frame->iter_dev;
  void *cond_dev = frame->cond_dev;
  int64_t *iter_cpu_buf = static_cast<int64_t *>(frame->iter_cpu_buf);
  int8_t *cond_host = static_cast<int8_t *>(frame->cond_host);
  hipStream_t stream = static_cast<hipStream_t>(state->stream);

  // Initialize cond_in for the body to read. cond_init is guaranteed true
  // here (false case short-circuited above). Host writes cond_host once,
  // before any kernel launches read cond_dev, so no host-vs-stream race
  // (the launch of iter 0 below already drains any prior writer).
  *cond_host = int8_t{1};

  void **current_descs = initial_descs;
  void **next_descs = scratch_descs_a;
  bool keep_going = true;
  int rc = 0;
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
      fprintf(stderr,
              "hipdnn_ep_run_*_loop: hipMemcpyAsync for iter[%lld] failed\n",
              static_cast<long long>(i));
      rc = -1;
      break;
    }
    frame->status = 0;
    for (int32_t carrier = 0; carrier < num_loop_carried; ++carrier) {
      frame->carriers[carrier].allocated_bank = -1;
      frame->carriers[carrier].published_bank = -2;
    }
    rc = body_fn(state, frame, iter_dev, cond_dev, current_descs, capture_descs,
                 next_descs);
    if (rc != 0)
      break;
    for (int32_t carrier = 0; carrier < num_loop_carried; ++carrier) {
      auto &banks = frame->carriers[carrier];
      if (banks.published_bank == -2) {
        rc = -1;
        break;
      }
    }
    if (rc != 0)
      break;
    if constexpr (CondPolicy::consultsCond()) {
      if (CondPolicy::checkCond(frame, cond_host, &keep_going) != 0) {
        rc = -1;
        break;
      }
    }
    current_descs = next_descs;
    next_descs =
        current_descs == scratch_descs_a ? scratch_descs_b : scratch_descs_a;
    for (int32_t carrier = 0; carrier < num_loop_carried; ++carrier) {
      auto &banks = frame->carriers[carrier];
      if (banks.published_bank >= 0)
        banks.current_bank = banks.published_bank;
    }
    *final_descs = current_descs;
  }
  if (rc != 0) {
    // The callback contract is transactional: current_descs was swapped only
    // after complete success, so final_descs still names the prior set.
    (void)hipdnn_ep_state_set_error_flag(state);
    *final_descs = initial_descs;
    if (synchronizeLoopStream(stream) == hipSuccess)
      destroyFrame(frame);
    else
      quarantineFrame(state, frame);
    return rc;
  }
  frame->parent = parent_frame;
  if (parent_frame) {
    frame->next_sibling = parent_frame->first_child;
    parent_frame->first_child = frame;
  }
  *frame_out = frame;
  return 0;
}

} // namespace

#ifdef HIPDNN_EP_RUNTIME_TESTING
extern "C" void hipdnn_ep_test_fail_next_loop_sync() {
  failNextLoopSync = true;
}
extern "C" void hipdnn_ep_test_fail_next_loop_alloc() {
  failNextLoopAlloc = true;
}
#endif

extern "C" int hipdnn_ep_run_counted_loop(
    RuntimeState *state, HipdnnEpLoopBodyFn body_fn, int64_t max_trip_count,
    bool cond_init, int32_t num_loop_carried, int32_t num_captures,
    void **initial_descs, void **scratch_descs_a, void **scratch_descs_b,
    void **capture_descs, HipdnnEpLoopFrame *parent_frame, void ***final_descs,
    HipdnnEpLoopFrame **frame_out) {
  return runLoopImpl<CountedCondPolicy>(
      state, body_fn, max_trip_count, cond_init, num_loop_carried, num_captures,
      initial_descs, scratch_descs_a, scratch_descs_b, capture_descs,
      parent_frame, final_descs, frame_out);
}

extern "C" int hipdnn_ep_run_loop(
    RuntimeState *state, HipdnnEpLoopBodyFn body_fn, int64_t max_trip_count,
    bool cond_init, int32_t num_loop_carried, int32_t num_captures,
    void **initial_descs, void **scratch_descs_a, void **scratch_descs_b,
    void **capture_descs, HipdnnEpLoopFrame *parent_frame, void ***final_descs,
    HipdnnEpLoopFrame **frame_out) {
  return runLoopImpl<DynamicCondPolicy>(
      state, body_fn, max_trip_count, cond_init, num_loop_carried, num_captures,
      initial_descs, scratch_descs_a, scratch_descs_b, capture_descs,
      parent_frame, final_descs, frame_out);
}

extern "C" void *hipdnn_ep_loop_frame_alloc(HipdnnEpLoopFrame *frame,
                                            int32_t carrier_index,
                                            const int64_t *shape, int64_t rank,
                                            int64_t elem_size) {
  if (!frame || carrier_index < 0 || carrier_index >= frame->num_carriers ||
      rank < 0 || elem_size <= 0 || (rank > 0 && !shape)) {
    if (frame)
      frame->status = -1;
    return nullptr;
  }

  size_t bytes = static_cast<size_t>(elem_size);
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] < 0) {
      frame->status = -1;
      return nullptr;
    }
    size_t dim = static_cast<size_t>(shape[i]);
    if (dim != 0 && bytes > std::numeric_limits<size_t>::max() / dim) {
      frame->status = -1;
      return nullptr;
    }
    bytes *= dim;
  }
  auto &carrier = frame->carriers[carrier_index];
  int bank = carrier.current_bank == 0 ? 1 : 0;
  carrier.allocated_bank = bank;
  if (bytes == 0)
    return nullptr;
  if (bytes > carrier.capacity[bank]) {
    if (carrier.data[bank]) {
      if (synchronizeLoopStream(
              static_cast<hipStream_t>(frame->state->stream)) != hipSuccess ||
          hipFree(carrier.data[bank]) != hipSuccess) {
        frame->status = -1;
        return nullptr;
      }
      carrier.data[bank] = nullptr;
      carrier.capacity[bank] = 0;
    }
#ifdef HIPDNN_EP_RUNTIME_TESTING
    if (failNextLoopAlloc) {
      failNextLoopAlloc = false;
      frame->status = -1;
      return nullptr;
    }
#endif
    if (hipMalloc(&carrier.data[bank], bytes) != hipSuccess) {
      frame->status = -1;
      return nullptr;
    }
    carrier.capacity[bank] = bytes;
  }
  return carrier.data[bank];
}

extern "C" int hipdnn_ep_loop_frame_status(HipdnnEpLoopFrame *frame) {
  return frame ? frame->status : -1;
}

extern "C" int hipdnn_ep_loop_frame_set_current(HipdnnEpLoopFrame *frame,
                                                int32_t carrier_index,
                                                const void *current_data) {
  if (!frame || carrier_index < 0 || carrier_index >= frame->num_carriers)
    return -1;
  frame->carriers[carrier_index].current_data = current_data;
  return 0;
}

extern "C" int hipdnn_ep_loop_frame_publish(HipdnnEpLoopFrame *frame,
                                            int32_t carrier_index,
                                            const void *published_data) {
  if (!frame || carrier_index < 0 || carrier_index >= frame->num_carriers)
    return -1;
  auto &carrier = frame->carriers[carrier_index];
  int bank = carrier.allocated_bank;
  if (bank >= 0 && published_data == carrier.data[bank]) {
    carrier.published_bank = bank;
    return 0;
  }
  if (published_data == carrier.current_data) {
    carrier.published_bank = -1;
    return 0;
  }
  frame->status = -1;
  return -1;
}

extern "C" int hipdnn_ep_loop_frame_destroy(RuntimeState *state,
                                            HipdnnEpLoopFrame *frame) {
  if (!frame)
    return 0;
  if (!state || frame->state != state)
    return -1;
  unlinkFromParent(frame);
  if (synchronizeLoopStream(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    quarantineFrame(state, frame);
    (void)hipdnn_ep_state_set_error_flag(state);
    return -1;
  }
  destroyFrame(frame);
  return 0;
}

extern "C" int hipdnn_ep_loop_state_init(RuntimeState *state) {
  if (!state)
    return -1;
  state->loop_frames_mutex = new std::mutex();
  state->quarantined_loop_frames = nullptr;
  return 0;
}

extern "C" void hipdnn_ep_loop_cleanup_quarantined_frames(RuntimeState *state) {
  if (!state)
    return;
  std::mutex *mutex = getFrameMutex(state);
  HipdnnEpLoopFrame *frames = nullptr;
  if (mutex) {
    std::lock_guard<std::mutex> lock(*mutex);
    frames = static_cast<HipdnnEpLoopFrame *>(state->quarantined_loop_frames);
    state->quarantined_loop_frames = nullptr;
  } else {
    frames = static_cast<HipdnnEpLoopFrame *>(state->quarantined_loop_frames);
    state->quarantined_loop_frames = nullptr;
  }
  while (frames) {
    HipdnnEpLoopFrame *next = frames->quarantine_next;
    destroyFrame(frames);
    frames = next;
  }
}

extern "C" void hipdnn_ep_loop_state_cleanup(RuntimeState *state) {
  if (!state)
    return;
  if (synchronizeLoopStream(static_cast<hipStream_t>(state->stream)) ==
      hipSuccess)
    hipdnn_ep_loop_cleanup_quarantined_frames(state);
  delete getFrameMutex(state);
  state->loop_frames_mutex = nullptr;
}
