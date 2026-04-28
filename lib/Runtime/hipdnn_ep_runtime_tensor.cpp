/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hip_cleanup.h"
#include "hipdnn_ep_runtime.h"
#include "operator_profile.h"
#include "runtime_state_internal.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Fallback element size when tensor metadata is missing (covers fp32).
static constexpr size_t kDefaultElementSize = 4;
//===----------------------------------------------------------------------===//
// Per-Inference Performance Measurement (gated on HIPDNN_EP_PERF)
//===----------------------------------------------------------------------===//
// Reports a wall-clock "total runtime time" reference per inference plus a
// GPU-stream breakdown (H2D / Compute / D2H). Useful for triangulating
// against TTFT/TPS at the benchmark level and against op/io profile sums.
//
// The two clocks bracket DIFFERENT intervals -- this is intentional and the
// gap between them is itself informative:
//
//   Wall = host_end - host_start   (host monotonic clock, chrono)
//     host_start: prepare_input(0) entry
//     host_end:   last finalize_output return -- i.e. the moment the EP
//                 finished QUEUEING all GPU work and returned to OGA.
//                 NOT the moment that work has FINISHED on the GPU.
//
//   GPU stream = elapsed(h2d_start, d2h_end)   (GPU timer, hipEventElapsedTime)
//     h2d_start: queued at prepare_input(0); fires when the GPU processes
//                the first H2D dispatch.
//     d2h_end:   re-recorded at every D2H finalize_output (last write wins);
//                fires when the GPU finishes the last D2H copy.
//
// Interpretation of (Wall - GPU):
//   > 0  "Other"   = host-side framework overhead the operator/io profilers
//                    cannot see (synchronous reads, dispatch, error checks,
//                    PERF/op-profile bookkeeping itself, etc.).
//   < 0  "Overlap" = host returned to OGA before GPU finished draining the
//                    stream. The magnitude is how much GPU work was still
//                    in flight after host_end -- i.e. how much the host
//                    pipelined ahead of the GPU. A persistent overlap is
//                    THE signature of a GPU-bound steady state: speeding
//                    up host-side work cannot help, only kernels can.
//
// Note that the OLD design anchored host_end to the NEXT prepare_input(0)
// on the same model. By construction that made host_end >= T_d2h_end, so
// "Other" was always >= 0 and frequently pinned to ~0 -- giving the false
// impression of "100% GPU bound, no host overlap". The new design samples
// host_end at the last finalize_output, which exposes real overlap.
//
// Per-inference event flow on the runtime stream:
//   prepare_input(0)        -> H2D start  (and Wall start, via chrono)
//   prepare_output(0)       -> H2D end / Compute start
//   first finalize_output   -> D2H start  (Compute end; idempotent)
//   every D2H finalize_*    -> D2H end re-recorded after the real D2H
//                              (last write wins; cache-claim path skipped
//                              to avoid stamping on an idle stream)
//   every finalize_output   -> host_end sampled (last write wins; both
//                              real-D2H AND cache-claim paths)
//   state_cleanup / atexit  -> drain queue, print aggregate per session
//
// CRITICAL: closing each inference at the LAST finalize_output (rather than
// at the next prepare_input(0)) is what makes Wall meaningful for sparse-
// call models. In OGA's chunked decoder pipeline the prompt model is only
// invoked once per benchmark iteration (~52 s apart), so anchoring
// host_end / d2h_end to "next prepare_input on this model" would inflate
// Wall by the entire decode phase between calls. Re-recording on every
// finalize_output is essentially free (each call costs one chrono read
// plus, on the real-D2H path, one hipEventRecord on a stream that is
// already busy with D2Hs) and naturally pins the close marker to the
// very last D2H completion of the inference.
//
// All HIP events are RECORDED inline (no synchronization) and their elapsed
// times are READ in a single batch at end of run. This deliberately avoids
// hipEventSynchronize per inference because that would force CPU/GPU
// serialization and distort the very Wall number we are trying to measure.
//
// Warmup handling: the first N inferences are bucketed separately so kernel-
// selection / autotuning / JIT / hipBLASLt heuristic-search cost is reported
// but does not pollute steady-state averages. Controlled by
// HIPDNN_EP_PERF_WARMUP=N (default 1).
//
// Env vars:
//   HIPDNN_EP_PERF=1       -> enable measurement, print aggregate at exit
//   HIPDNN_EP_PERF_EACH=1  -> additionally print per-inference rows (verbose)
//   HIPDNN_EP_PERF_WARMUP=N -> bucket first N inferences as warmup (default 1)

struct PerfPending {
  // GPU phase markers (all on the same stream).
  hipEvent_t h2d_start = nullptr;
  hipEvent_t h2d_end = nullptr;
  hipEvent_t d2h_start = nullptr;
  hipEvent_t d2h_end = nullptr;
  // Host wall-clock bracket. host_start is sampled at prepare_input(0);
  // host_end is sampled at every finalize_output (last write wins). If no
  // finalize_output was called, perf_close_current_inference() falls back
  // to the late-close path.
  std::chrono::steady_clock::time_point host_start{};
  std::chrono::steady_clock::time_point host_end{};
  bool host_end_set = false;
  // Per-inference IO sizes seen by prepare_input / finalize_output. Used to
  // sanity-check against the IO profile and to populate aggregate stats.
  size_t h2d_bytes = 0;
  size_t h2d_count = 0;
  size_t d2h_bytes = 0;
  size_t d2h_count = 0;
};

struct PerfAggregate {
  unsigned count = 0;
  double wall_ms = 0.0;
  double h2d_ms = 0.0;
  double compute_ms = 0.0;
  double d2h_ms = 0.0;
  size_t h2d_count = 0;
  size_t h2d_bytes = 0;
  size_t d2h_count = 0;
  size_t d2h_bytes = 0;
};

// Pool of recyclable hipEvents to avoid hipEventCreate/Destroy churn.
static std::vector<hipEvent_t> g_perf_event_pool;
// Closed inferences waiting to have their event timings read at end of run.
static std::vector<PerfPending> g_perf_queue;
// The currently-open inference being recorded in real time.
static PerfPending g_perf_current;
static bool g_perf_current_valid = false;
// Stashed for the atexit dump, since we no longer get a RuntimeState there.
static void *g_perf_last_stream = nullptr;
// Per-session aggregates: separate buckets for warmup vs steady-state so
// the kernel-selection / JIT cost of the first inferences doesn't pollute
// the steady-state averages. See HIPDNN_EP_PERF_WARMUP.
//
// These are RESET after each perf_flush_and_print() so chunked-pipeline
// models -- where each ONNX session (prompt model, decoder model, ...)
// triggers its own state_cleanup -- produce one PERF block per session
// rather than one cumulative blob across the whole process.
static PerfAggregate g_perf_warmup_agg;
static PerfAggregate g_perf_steady_agg;
// Counts how many state_cleanup / atexit prints have happened. Used to
// label each PERF block ("session #1", "session #2", ...) so the user can
// match them up against OGA's prompt vs decoder model.
static unsigned g_perf_session_index = 0;

static bool hipdnn_ep_perf_each_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_PERF_EACH");
    return v && v[0] >= '1';
  }();
  return enabled;
}

// hipdnn_ep_perf_warmup_count() lives in debug_log.h so OP_PROFILE can use
// the same value -- both reports must partition warmup vs steady identically
// or their averages won't be directly comparable.

static hipEvent_t perf_acquire_event() {
  if (!g_perf_event_pool.empty()) {
    hipEvent_t e = g_perf_event_pool.back();
    g_perf_event_pool.pop_back();
    return e;
  }
  hipEvent_t e = nullptr;
  if (hipEventCreate(&e) != hipSuccess)
    return nullptr;
  return e;
}

static void perf_release_event(hipEvent_t e) {
  if (e)
    g_perf_event_pool.push_back(e);
}

// Forward declarations for the rollup helpers.
static void perf_close_current_inference(void *stream);
static void perf_drain_and_aggregate();
static void perf_print_aggregate();
static void perf_atexit_dump();

// Begin a new pending inference at prepare_input(index=0). If an inference
// is still open (e.g., something happened that prevented finalize_output
// from running, or this is the first call into a new model), close it first.
static void perf_begin_inference(void *stream) {
  if (!hipdnn_ep_perf_enabled())
    return;
  g_perf_last_stream = stream;

  // If the previous inference is still open here, finalize_output never
  // closed it (rare). Close it now via the late-close path.
  if (g_perf_current_valid)
    perf_close_current_inference(stream);

  // Allocate a fresh set of events for the new inference.
  g_perf_current = PerfPending{};
  g_perf_current.h2d_start = perf_acquire_event();
  g_perf_current.h2d_end = perf_acquire_event();
  g_perf_current.d2h_start = perf_acquire_event();
  g_perf_current.d2h_end = perf_acquire_event();
  g_perf_current.host_start = std::chrono::steady_clock::now();
  g_perf_current_valid = true;

  // Register the atexit dump exactly once on the first PERF-enabled call.
  static bool atexit_registered = false;
  if (!atexit_registered) {
    std::atexit(perf_atexit_dump);
    atexit_registered = true;
  }

  if (g_perf_current.h2d_start)
    (void)hipEventRecord(g_perf_current.h2d_start,
                         static_cast<hipStream_t>(stream));
}

static void perf_record_h2d_end(void *stream) {
  if (!g_perf_current_valid)
    return;
  if (g_perf_current.h2d_end)
    (void)hipEventRecord(g_perf_current.h2d_end,
                         static_cast<hipStream_t>(stream));
}

// Idempotent: only the FIRST finalize_output of an inference actually
// records the d2h_start marker. Subsequent calls just return.
static void perf_record_d2h_start(void *stream) {
  if (!g_perf_current_valid)
    return;
  if (g_perf_current.d2h_count != 0)
    return;
  if (g_perf_current.d2h_start)
    (void)hipEventRecord(g_perf_current.d2h_start,
                         static_cast<hipStream_t>(stream));
}

static void perf_count_h2d(size_t bytes) {
  if (!g_perf_current_valid)
    return;
  g_perf_current.h2d_bytes += bytes;
  g_perf_current.h2d_count++;
}

static void perf_count_d2h(size_t bytes) {
  if (!g_perf_current_valid)
    return;
  g_perf_current.d2h_bytes += bytes;
  g_perf_current.d2h_count++;
}

// Sample host_end (last write wins). Cheap (~100 ns chrono read), called
// from every finalize_output -- including the IoCache fast path that
// returns without touching the stream -- so it stays accurate regardless
// of whether the LAST finalize did a real D2H or just a cache claim.
static inline void perf_touch_host_end() {
  if (!g_perf_current_valid)
    return;
  g_perf_current.host_end = std::chrono::steady_clock::now();
  g_perf_current.host_end_set = true;
}

// Re-record d2h_end on the stream (last write wins). Only called from the
// finalize path that actually queued an async D2H -- recording on an idle
// stream would back-fill d2h_end with a timestamp from "much later than
// the actual last D2H" and inflate d2h_ms by stream-idle time. By
// re-recording right after each real hipMemcpyAsync, the FINAL stored
// timestamp is exactly "right after the last D2H of the inference".
//
// IMPORTANT: this is what avoids the prompt-model-Wall=52s bug. Anchoring
// the close to "next prepare_input on the same model" fails for sparse
// callers (prompt model called once per benchmark iteration with 255
// decode calls in between).
static inline void perf_record_d2h_end(void *stream) {
  if (!g_perf_current_valid)
    return;
  if (g_perf_current.d2h_end && stream)
    (void)hipEventRecord(g_perf_current.d2h_end,
                         static_cast<hipStream_t>(stream));
}

// Late-close path: only used when the current inference somehow reaches a
// teardown boundary (next prepare_input on a sparse model, state_cleanup,
// atexit) without finalize_output ever having closed it. In that case we
// stamp the close markers now -- accuracy is degraded for those rare
// inferences (Wall will include framework idle time after the last call)
// but at least the data is not lost.
static void perf_close_current_inference(void *stream) {
  if (!g_perf_current_valid)
    return;
  if (!g_perf_current.host_end_set) {
    if (g_perf_current.d2h_end && stream)
      (void)hipEventRecord(g_perf_current.d2h_end,
                           static_cast<hipStream_t>(stream));
    g_perf_current.host_end = std::chrono::steady_clock::now();
    g_perf_current.host_end_set = true;
  }
  g_perf_queue.push_back(g_perf_current);
  g_perf_current = PerfPending{};
  g_perf_current_valid = false;
}

// Add a pending inference to the right aggregate bucket and (if EACH is on)
// emit a per-inference row.
static void perf_aggregate_one(unsigned idx, const PerfPending &p,
                               double wall_ms, float h2d_ms, float compute_ms,
                               float d2h_ms) {
  PerfAggregate &agg = (idx < hipdnn_ep_perf_warmup_count())
                           ? g_perf_warmup_agg
                           : g_perf_steady_agg;
  agg.count++;
  agg.wall_ms += wall_ms;
  agg.h2d_ms += h2d_ms;
  agg.compute_ms += compute_ms;
  agg.d2h_ms += d2h_ms;
  agg.h2d_count += p.h2d_count;
  agg.h2d_bytes += p.h2d_bytes;
  agg.d2h_count += p.d2h_count;
  agg.d2h_bytes += p.d2h_bytes;

  if (hipdnn_ep_perf_each_enabled()) {
    double gpu_ms = static_cast<double>(h2d_ms) +
                    static_cast<double>(compute_ms) +
                    static_cast<double>(d2h_ms);
    // delta = Wall - GPU. Sign matters:
    //   delta > 0 -> "Other": host-side overhead the profilers can't see
    //                (sync, framework, dispatch).
    //   delta < 0 -> "Overlap": host returned before GPU finished. The
    //                magnitude is how much GPU work was still in flight
    //                after host_end (i.e. how much the host pipelined
    //                ahead of the GPU). Indicates a GPU-bound steady state.
    double delta_ms = wall_ms - gpu_ms;
    fprintf(stderr,
            "[PERF] inference #%u%s:  Wall %.3f ms  (GPU %.3f ms = "
            "H2D %.3f + Compute %.3f + D2H %.3f, %s %.3f)\n",
            idx + 1, idx < hipdnn_ep_perf_warmup_count() ? " [warmup]" : "",
            wall_ms, gpu_ms, (double)h2d_ms, (double)compute_ms, (double)d2h_ms,
            delta_ms >= 0.0 ? "Other" : "Overlap",
            delta_ms >= 0.0 ? delta_ms : -delta_ms);
  }
}

// At end of run, sync once on the very last d2h_end event, walk the queue,
// query elapsed times, partition into warmup / steady aggregates, recycle
// events.
static void perf_drain_and_aggregate() {
  if (g_perf_queue.empty())
    return;

  // Single sync point: wait for all queued GPU work to drain so all event
  // timestamps are valid.
  hipEvent_t last_end = g_perf_queue.back().d2h_end;
  if (last_end)
    (void)hipEventSynchronize(last_end);

  for (size_t i = 0; i < g_perf_queue.size(); ++i) {
    auto &p = g_perf_queue[i];
    double wall_ms =
        std::chrono::duration<double, std::milli>(p.host_end - p.host_start)
            .count();
    if (wall_ms < 0.0)
      wall_ms = 0.0;

    float h2d_ms = 0.0f, compute_ms = 0.0f, d2h_ms = 0.0f;
    if (p.h2d_start && p.h2d_end)
      (void)hipEventElapsedTime(&h2d_ms, p.h2d_start, p.h2d_end);
    if (p.h2d_end && p.d2h_start)
      (void)hipEventElapsedTime(&compute_ms, p.h2d_end, p.d2h_start);
    if (p.d2h_start && p.d2h_end)
      (void)hipEventElapsedTime(&d2h_ms, p.d2h_start, p.d2h_end);
    if (h2d_ms < 0.0f)
      h2d_ms = 0.0f;
    if (compute_ms < 0.0f)
      compute_ms = 0.0f;
    if (d2h_ms < 0.0f)
      d2h_ms = 0.0f;

    perf_aggregate_one(static_cast<unsigned>(i), p, wall_ms, h2d_ms, compute_ms,
                       d2h_ms);

    // Recycle events back to the pool for reuse.
    perf_release_event(p.h2d_start);
    perf_release_event(p.h2d_end);
    perf_release_event(p.d2h_start);
    perf_release_event(p.d2h_end);
  }
  g_perf_queue.clear();
}

// Print one aggregate block. Internal helper used by perf_print_aggregate().
static void perf_print_one_aggregate(const char *label,
                                     const PerfAggregate &a) {
  if (a.count == 0)
    return;
  double n = static_cast<double>(a.count);
  double gpu_total = a.h2d_ms + a.compute_ms + a.d2h_ms;
  // Wall - GPU. Sign carries information:
  //   delta > 0 -> host-side framework overhead the op/io profilers can't see
  //                (sync points, dispatch, sampling, etc.). Reported as
  //                "Other (Wall - GPU)".
  //   delta < 0 -> host returned to OGA before the GPU finished draining the
  //                stream. Magnitude is how much the host was pipelined ahead
  //                of the GPU on average. Reported as "Overlap (GPU - Wall)".
  //                A persistent overlap indicates the workload is GPU-bound:
  //                speeding up host-side work won't help.
  double delta_total = a.wall_ms - gpu_total;
  double mb_h2d = static_cast<double>(a.h2d_bytes) / (1024.0 * 1024.0);
  double mb_d2h = static_cast<double>(a.d2h_bytes) / (1024.0 * 1024.0);

  // Use Wall as the percentage base for "Other" (host overhead share of Wall)
  // and GPU as the base for "Overlap" (how much GPU work overlapped with the
  // *next* inference's host work, i.e. is hidden by pipelining).
  const char *delta_label =
      delta_total >= 0.0 ? "Other (Wall - GPU)" : "Overlap (GPU - Wall)";
  double delta_abs = delta_total >= 0.0 ? delta_total : -delta_total;
  double delta_pct_base = delta_total >= 0.0 ? a.wall_ms : gpu_total;
  double delta_pct =
      delta_pct_base > 0.0 ? (delta_abs / delta_pct_base * 100.0) : 0.0;

  fprintf(stderr,
          "[PERF] %s over %u inference(s):\n"
          "  Wall (CPU clock):       %10.2f ms total / %8.3f ms avg\n"
          "  GPU stream (H2D+C+D2H): %10.2f ms total / %8.3f ms avg "
          "(%.1f%% of Wall)\n"
          "    H2D:                  %10.2f ms total / %8.3f ms avg "
          "(%zu tensors, %.1f MB)\n"
          "    Compute (kernels):    %10.2f ms total / %8.3f ms avg\n"
          "    D2H:                  %10.2f ms total / %8.3f ms avg "
          "(%zu tensors, %.1f MB)\n"
          "  %-22s  %10.2f ms total / %8.3f ms avg (%.1f%%)\n",
          label, a.count, a.wall_ms, a.wall_ms / n, gpu_total, gpu_total / n,
          a.wall_ms > 0 ? (gpu_total / a.wall_ms * 100.0) : 0.0, a.h2d_ms,
          a.h2d_ms / n, a.h2d_count, mb_h2d, a.compute_ms, a.compute_ms / n,
          a.d2h_ms, a.d2h_ms / n, a.d2h_count, mb_d2h, delta_label, delta_abs,
          delta_abs / n, delta_pct);
}

// Print the per-session aggregate(s) and reset for the next session.
//
// We label this print with a session index so the user can match each PERF
// block to the corresponding ONNX session in OGA's chunked pipeline (e.g.,
// session #1 = prompt model, session #2 = decoder model). After printing,
// the aggregates are zeroed -- subsequent inferences (from a different
// session whose state_cleanup has not yet been called) accumulate fresh.
static void perf_print_aggregate() {
  if (g_perf_warmup_agg.count == 0 && g_perf_steady_agg.count == 0)
    return;

  ++g_perf_session_index;
  char header[128];
  std::snprintf(header, sizeof(header),
                "session #%u WARMUP (first inferences, kernel selection / "
                "JIT / autotune)",
                g_perf_session_index);
  perf_print_one_aggregate(header, g_perf_warmup_agg);
  std::snprintf(header, sizeof(header),
                "session #%u STEADY-STATE (post-warmup)", g_perf_session_index);
  perf_print_one_aggregate(header, g_perf_steady_agg);

  g_perf_warmup_agg = PerfAggregate{};
  g_perf_steady_agg = PerfAggregate{};
}

// Process-exit handler: best-effort drain of the queue and aggregate print.
static void perf_atexit_dump() {
  if (!hipdnn_ep_perf_enabled())
    return;
  if (g_perf_current_valid && g_perf_last_stream)
    perf_close_current_inference(g_perf_last_stream);
  perf_drain_and_aggregate();
  perf_print_aggregate();
}

// Public entry point for state_cleanup. Drains the queue and prints. The
// caller is expected to have already called hipStreamSynchronize(stream)
// before invoking this so the events are guaranteed to be valid.
extern "C" void hipdnn_ep_perf_flush_and_print(RuntimeState *state) {
  if (!hipdnn_ep_perf_enabled())
    return;
  if (g_perf_current_valid && state)
    perf_close_current_inference(state->stream);
  perf_drain_and_aggregate();
  perf_print_aggregate();
}

//===----------------------------------------------------------------------===//
// GPU Tensor Buffer Pool
//===----------------------------------------------------------------------===//
// Reuses GPU allocations across inferences to eliminate per-call
// hipMalloc/hipFree overhead. Safe because tensor shapes (and therefore sizes)
// are fixed across inferences for a given model.

static std::unordered_map<size_t, std::vector<void *>> g_gpu_buffer_pool;

static void *pool_alloc(size_t size_bytes) {
  assert(size_bytes > 0 && "pool_alloc: size_bytes must be positive");
  auto it = g_gpu_buffer_pool.find(size_bytes);
  if (it != g_gpu_buffer_pool.end() && !it->second.empty()) {
    void *ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }
  void *ptr = nullptr;
  if (hipMalloc(&ptr, size_bytes) != hipSuccess)
    return nullptr;
  return ptr;
}

static void pool_release(void *ptr, size_t size_bytes) {
  assert(size_bytes > 0 && "pool_release: size_bytes must be positive");
  if (ptr)
    g_gpu_buffer_pool[size_bytes].push_back(ptr);
}

//===----------------------------------------------------------------------===//
// I/O Cache: skip H2D + (steady-decode) D2H for rank-4 KV tensors
//===----------------------------------------------------------------------===//
// CORRECT VARIANT (PIN+selective-D2H, rank-4 only, per-state).
//
// OGA with past_present_share_buffer=true passes the same host OrtValue
// as both the "past_key_N" input and the "present_key_N" output. Without
// a cache the runtime copies that buffer H2D at every prepare_input step.
// For Mistral-7B at ctx=16384 that is ~2 GiB of redundant PCIe traffic
// per token.
//
// Design constraints for this variant:
//   1. Cache only rank-4 tensors. OGA's KV cache tensors are shaped
//      [batch, num_kv_heads, seq_len, head_size] (rank 4); all other
//      inputs (input_ids, attention_mask, position_ids) are rank <=2 and
//      intentionally not cached -- they change every step and their host
//      contents must be read every time.
//   2. Per-state. Each RuntimeState owns its own map; prefill's cache
//      and decode's cache do not share. In OGA's chunked decoder
//      pipeline prefill and decode load distinct MLIR DLLs so the caches
//      would already be separate via file-scope statics, but keeping the
//      cache inside the state makes that intent explicit (and still
//      works correctly if multiple states share a DLL).
//   3. First use reads from host (CPU source of truth). The cache
//      installs a GPU buffer only on cache_claim (input/output alias);
//      a session that has not yet seen the host_ptr MISSes, pool_allocs,
//      and H2Ds.
//
// Correctness invariant: D2H runs whenever the host buffer is the
// downstream source of truth, and H2D runs whenever the host buffer may
// have been mutated between calls. Concretely:
//   * Prefill calls ALWAYS do both H2D and D2H on rank-4 cache_claim
//     outputs (cache serves only as an allocation reuse mechanism --
//     no bandwidth savings). Rationale: in chat mode the prompt is
//     split into multiple prefill chunks that may span many turns, and
//     OGA can freely mutate the KV host buffers between chunks (KV
//     trimming, sliding-window rotation, new-turn prompt ingestion).
//     Skipping either direction would desync host and GPU.
//   * Decode calls skip H2D on rank-4 cache HITs (GPU is authoritative
//     in steady state) and skip D2H when every rank-4 input was a HIT
//     (the session hands KV off to itself via the cache, no external
//     observer reads the host buffer). The first decode call of each
//     generator is all-MISS (gen-boundary GC cleared the previous
//     generator's entries), so H2D runs to pull prefill's output into
//     the cache and D2H runs to preserve the invariant cheaply.
//
// The net effect: prefill -> decode handoff, chunked-prefill correctness,
// and multi-turn chat all stay correct; logits are identical to the no-
// cache path; steady-state decode still avoids ~2 GiB of H2D+D2H
// traffic per token.
//
// Prefill vs decode detection: at prepare_input(idx==0) we pick the
// smallest last-dim across rank>=2 non-KV inputs. Decode always has a
// rank>=2 input with last_dim == 1 (input_ids / position_ids carry one
// new token); prefill's corresponding inputs carry the chunk so their
// last_dim is chunk_size > 1. This is model-agnostic for GenAI-style
// decoder pipelines; ambiguous cases default to prefill (the safe
// choice -- just turns off the skip, no correctness risk).
//
// Operations:
//   - prepare_input(idx==0):
//       * Build the alive set (current_inputs) from ALL of this call's
//         rank-4 inputs. Non-rank-4 inputs are not tracked since they
//         are never cached.
//       * Generation-boundary GC: any cache entry whose host_ptr is NOT
//         in current_inputs is stale (the previous OgaGenerator's KV
//         buffers were freed). hipStreamSynchronize and hipFree them so
//         VRAM actually returns to the OS before this call allocates.
//       * Mode detection: is_prefill := min rank>=2 non-KV last_dim > 1.
//       * Steady-decode detection: skip_rank4_d2h := !is_prefill AND
//         every rank-4 input is already in the cache (all HITs).
//         Consumed by finalize_output.
//   - prepare_input (per-tensor):
//       * If tensor->rank != 4: skip the cache, pool_alloc + H2D as
//         before.
//       * If tensor->rank == 4: look up host_ptr in this state's cache.
//           HIT in decode : reuse gpu_ptr, skip H2D.
//           HIT in prefill: reuse gpu_ptr, RUN H2D (host may have
//                           changed between chunks / turns).
//           MISS          : pool_alloc + H2D.
//   - finalize_output:
//       * cache_claim = (rank == 4) && (host_ptr in current_inputs).
//       * Skip D2H when cache_claim && skip_rank4_d2h (decode steady
//         state only); otherwise run it. Prefill forces D2H via
//         skip_rank4_d2h=false.
//       * On cache_claim install (host_ptr -> gpu_ptr) in this state's
//         cache. Otherwise pool_release.
//   - free_input:
//       * If this state's cache now holds (host_ptr, gpu_ptr), leave it
//         alone. Otherwise pool_release.
//   - state_cleanup:
//       * hipFree every entry's GPU buffer (stream was already synced by
//         hipdnn_ep_state_cleanup).
//
// Gate: HIPDNN_EP_IO_CACHE (default=ON, set to "0" to disable for A/B
// comparisons against the no-cache path).

struct IoCacheEntry {
  void *gpu_ptr;
  size_t size_bytes;
};

// Per-state cache. Keyed by host_ptr; one map per RuntimeState.
struct IoCache {
  std::unordered_map<const void *, IoCacheEntry> entries;
  // Alive set of rank-4 input host pointers for the in-flight call.
  // Refreshed at prepare_input(idx==0) and consumed by finalize_output to
  // detect input/output aliasing (cache_claim).
  std::unordered_set<const void *> current_inputs;
  // True when this call looks like prefill (multi-token input chunk).
  // Determined at prepare_input(idx==0) from input shapes: decode always
  // has at least one rank>=2 non-KV input with last_dim == 1 (the single
  // new token), while prefill carries the chunk so min last_dim > 1.
  // Prefill forces both H2D (host may have been mutated since last
  // call) and D2H (hand off correct KV to the next prefill chunk /
  // decode / turn) on rank-4 cache_claim outputs. Default is prefill
  // (safe: just turns off the skip, no correctness risk).
  bool is_prefill = true;
  // True when EVERY rank-4 input of this call was already in the cache
  // (all HITs, post gen-boundary GC) AND this call is decode. That only
  // happens once this state has been running its own decode loop --
  // every prefill call, the first call of every new generator, and any
  // call with a fresh rank-4 input leave this false. When true,
  // finalize_output can skip D2H for rank-4 cache_claim outputs: the
  // decode session hands off to itself via the cache, and no external
  // observer reads the host KV buffer in that steady state, so keeping
  // the host buffer in sync is pure bandwidth waste. Computed at
  // prepare_input(idx==0).
  bool skip_rank4_d2h = false;
};

static bool io_cache_enabled() {
  // Read once at process startup, log the resolved state to stderr so it
  // is visible in benchmark logs alongside other runtime toggles.
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_IO_CACHE");
    bool on = !v || v[0] != '0';
    fprintf(stderr,
            "[Runtime] HIPDNN_EP_IO_CACHE=%s (I/O cache %s, "
            "mode=pin+d2h, rank-4 only, per-state)\n",
            v ? v : "<unset>", on ? "ENABLED" : "DISABLED");
    return on;
  }();
  return enabled;
}

static IoCache *io_cache_get(RuntimeState *state) {
  if (!state)
    return nullptr;
  if (!state->io_cache)
    state->io_cache = new IoCache();
  return static_cast<IoCache *>(state->io_cache);
}

// Look up a host_ptr in this state's cache. Returns the entry on HIT; on
// size mismatch, evicts the stale entry (hipFree) and returns nullptr
// (the caller will pool_alloc + H2D).
static const IoCacheEntry *io_cache_lookup(IoCache *cache, const void *host_ptr,
                                           size_t size_bytes) {
  if (!cache || !host_ptr)
    return nullptr;
  auto it = cache->entries.find(host_ptr);
  if (it == cache->entries.end())
    return nullptr;
  if (it->second.size_bytes != size_bytes) {
    RUNTIME_DEBUG_LOG(
        "[Runtime DEBUG] io_cache EVICT (size change) host=%p gpu=%p "
        "old_size=%zu new_size=%zu\n",
        host_ptr, it->second.gpu_ptr, it->second.size_bytes, size_bytes);
    hipFree(it->second.gpu_ptr);
    cache->entries.erase(it);
    return nullptr;
  }
  return &it->second;
}

// Install/overwrite a cache entry in this state's cache. The OLD gpu_ptr
// (if any) is NOT released here: it is still referenced by THIS call's
// input TensorBuffer at install time, and free_input releases it (its
// gpu_ptr no longer matches the cache's new entry).
static void io_cache_install(IoCache *cache, const void *host_ptr,
                             void *gpu_ptr, size_t size_bytes) {
  if (!cache)
    return;
  cache->entries[host_ptr] = IoCacheEntry{gpu_ptr, size_bytes};
}

// True iff this state's cache currently holds gpu_ptr under host_ptr.
// Used by free_input to decide whether the cache still owns the buffer
// (skip release) or has moved on (pool_release).
static bool io_cache_owns(IoCache *cache, const void *host_ptr,
                          const void *gpu_ptr) {
  if (!cache || !host_ptr || !gpu_ptr)
    return false;
  auto it = cache->entries.find(host_ptr);
  return it != cache->entries.end() && it->second.gpu_ptr == gpu_ptr;
}

// Invoked from hipdnn_ep_state_cleanup via runtime_state_internal.h.
// Releases every cached GPU buffer. hipFree (not pool_release) because
// at state teardown there is no later consumer within this DLL; we want
// VRAM to return to the OS, not sit in an orphaned pool.
//
// Caller MUST have synchronized this state's stream first so no
// in-flight kernel still references the buffers. (See
// hipdnn_ep_state_cleanup which calls hipStreamSynchronize before us.)
void hipdnn_ep_io_cache_destroy(RuntimeState *state) {
  if (!state || !state->io_cache)
    return;
  auto *cache = static_cast<IoCache *>(state->io_cache);
  for (const auto &kv : cache->entries) {
    RUNTIME_DEBUG_LOG("[Runtime DEBUG] io_cache GC host=%p gpu=%p size=%zu "
                      "(state destroyed)\n",
                      kv.first, kv.second.gpu_ptr, kv.second.size_bytes);
    hipFree(kv.second.gpu_ptr);
  }
  delete cache;
  state->io_cache = nullptr;
}

// Element size is read from tensor_t.element_size (set by EP caller)

// Helper function to check and log gcnArchName
static void check_gcnarch(const char *location) {
  if (!hipdnn_ep_debug_enabled())
    return;
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err == hipSuccess) {
    fprintf(stderr, "[%s] gcnArchName='%s' (len=%zu)\n", location,
            prop.gcnArchName, strlen(prop.gcnArchName));
  } else {
    fprintf(stderr, "[%s] ERROR: hipGetDeviceProperties failed: %d\n", location,
            err);
  }
}

// Helper: Calculate total size in bytes for a tensor
// Returns 0 on error (overflow or invalid dimensions)
static size_t calculateTensorSize(const int64_t *shape, size_t rank,
                                  size_t element_size) {
  if (rank == 0) {
    return element_size; // Rank-0 scalar: 1 element
  }
  if (!shape) {
    return 0;
  }

  // Validate all dimensions are positive
  for (size_t i = 0; i < rank; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "Invalid dimension at index %zu: %lld\n", i,
              (long long)shape[i]);
      return 0;
    }
  }

  // Calculate total number of elements with overflow check
  size_t total_elements = 1;
  for (size_t i = 0; i < rank; i++) {
    if (total_elements > SIZE_MAX / static_cast<size_t>(shape[i])) {
      fprintf(stderr, "Tensor size overflow at dimension %zu\n", i);
      return 0;
    }
    total_elements *= static_cast<size_t>(shape[i]);
  }

  if (total_elements > SIZE_MAX / element_size) {
    fprintf(stderr, "Tensor size overflow when applying element size\n");
    return 0;
  }

  return total_elements * element_size;
}

// Prepare input tensor: parse, validate, allocate GPU buffer, H2D transfer
int hipdnn_ep_tensor_prepare_input(RuntimeState *state, span_t *inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer *out_buffer) {
  check_gcnarch("BEFORE prepare_input");

  // VERIFICATION: Struct sizes
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] === Struct Size Verification ===\n");
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] sizeof(TensorBuffer) = %zu\n",
                    sizeof(TensorBuffer));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, gpu_ptr) = %zu\n",
                    offsetof(TensorBuffer, gpu_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, host_ptr) = %zu\n",
                    offsetof(TensorBuffer, host_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, shape_ptr) = %zu\n",
                    offsetof(TensorBuffer, shape_ptr));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, rank) = %zu\n",
                    offsetof(TensorBuffer, rank));
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] offsetof(TensorBuffer, size_bytes) = %zu\n",
      offsetof(TensorBuffer, size_bytes));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(TensorBuffer, is_pooled) = %zu\n",
                    offsetof(TensorBuffer, is_pooled));

  // VERIFICATION: tensor_t struct
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] sizeof(tensor_t) = %zu\n",
                    sizeof(tensor_t));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, data) = %zu\n",
                    offsetof(tensor_t, data));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, shape) = %zu\n",
                    offsetof(tensor_t, shape));
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] offsetof(tensor_t, rank) = %zu\n",
                    offsetof(tensor_t, rank));

  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!inputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null inputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_input: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // VERIFICATION: span_t access
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs pointer = %p\n", (void *)inputs);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs->data = %p\n",
                    (void *)inputs->data);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] inputs->count = %zu\n", inputs->count);

  // Validate index bounds
  if (index >= inputs->count) {
    fprintf(
        stderr,
        "hipdnn_ep_tensor_prepare_input: index %zu out of bounds (count=%zu)\n",
        index, inputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &inputs->data[index];

  // DUMP: Raw memory of tensor_t struct
  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] tensor_t struct memory dump (address=%p):\n",
      (void *)tensor);
  auto *bytes = reinterpret_cast<unsigned char *>(tensor);
  for (size_t i = 0; i < sizeof(tensor_t); i++) {
    RUNTIME_DEBUG_LOG("  [%02zu] = 0x%02x\n", i, bytes[i]);
  }

  // DUMP: Field values
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->data = %p\n", tensor->data);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->shape = %p\n",
                    (void *)tensor->shape);
  RUNTIME_DEBUG_LOG("[Runtime DEBUG] tensor->rank = %zu\n", tensor->rank);

  // Validate field access doesn't corrupt memory (re-read test)
  void *data_before = tensor->data;
  int64_t *shape_before = tensor->shape;
  size_t rank_before = tensor->rank;

  // Re-read and compare
  if (tensor->data != data_before || tensor->shape != shape_before ||
      tensor->rank != rank_before) {
    fprintf(stderr, "[Runtime ERROR] Struct fields changed on re-read!\n");
    fprintf(stderr, "  data: %p -> %p\n", data_before, tensor->data);
    fprintf(stderr, "  shape: %p -> %p\n", (void *)shape_before,
            (void *)tensor->shape);
    fprintf(stderr, "  rank: %zu -> %zu\n", rank_before, tensor->rank);
    return HIPDNN_EP_ERR_NULL_POINTER; // Use generic error code
  }

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!tensor->shape && tensor->rank != 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: rank mismatch at index %zu "
            "(expected %zu, got %zu)\n",
            index, expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Read element size from tensor struct (set by EP caller)
  size_t element_size = tensor->element_size;
  if (element_size == 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_input: tensor[%zu].element_size is 0, "
            "defaulting to %zu\n",
            index, kDefaultElementSize);
    element_size = kDefaultElementSize;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_input[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu\n",
      index, tensor->rank, element_size, size_bytes);

  // PERF: close the previous inference (queues d2h_end on the stream) and
  // open a new one. No synchronization here -- elapsed times are read in a
  // single batch at end of run by perf_drain_and_aggregate(), so this hot
  // path stays fully asynchronous and doesn't distort the Wall measurement.
  if (index == 0)
    perf_begin_inference(state->stream);

  // OP_PROFILE: flush previous inference's per-operator events at the start
  // of a new inference. Generated inference_compute() does not currently call
  // hipdnn_ep_stream_sync(), so we drive the flush from the input-prep phase.
  if (index == 0) {
    hipdnn_ep::op_profile_flush_inference();
  }

  // I/O cache plumbing (past_present_share_buffer fast-path, rank-4 only).
  // On the first input of a call, refresh the alive set from THIS call's
  // rank-4 inputs so finalize_output can detect cache_claim (input/output
  // aliasing) and so the generation-boundary sweep below knows which
  // host_ptrs are still live.
  IoCache *io_cache = io_cache_enabled() ? io_cache_get(state) : nullptr;
  if (io_cache && index == 0) {
    io_cache->current_inputs.clear();
    io_cache->current_inputs.reserve(inputs->count);
    for (size_t i = 0; i < inputs->count; ++i) {
      const tensor_t &t = inputs->data[i];
      if (t.data && t.rank == 4) {
        io_cache->current_inputs.insert(t.data);
      }
    }

    // Generation-boundary GC: when OGA recreates a generator the KV host
    // buffers from the previous generator are freed and a fresh set is
    // passed in. The cache still holds 32 entries (per rank-4 input)
    // keyed by the now-freed host pointers, each pinning a 32 MiB GPU
    // buffer. Without this sweep those entries accumulate every
    // iteration (~1 GiB per session per iteration) and OOM a few iters
    // in.
    //
    // Any cache entry whose host_ptr is NOT in this call's rank-4 alive
    // set is stale: hipFree the GPU buffer (return VRAM to the OS; the
    // per-size-class pool would just hold it indefinitely) and erase
    // the bookkeeping. We must hipStreamSynchronize first because the
    // previous call's main_graph may still be writing to the cached
    // output buffer when we enter this new call.
    std::vector<const void *> stale_keys;
    for (const auto &kv : io_cache->entries) {
      if (io_cache->current_inputs.count(kv.first) == 0) {
        stale_keys.push_back(kv.first);
      }
    }
    if (!stale_keys.empty()) {
      if (state->stream) {
        hipStreamSynchronize(static_cast<hipStream_t>(state->stream));
      }
      for (const void *key : stale_keys) {
        auto it = io_cache->entries.find(key);
        if (it != io_cache->entries.end()) {
          RUNTIME_DEBUG_LOG(
              "[Runtime DEBUG] io_cache GC (gen boundary) host=%p gpu=%p "
              "size=%zu\n",
              key, it->second.gpu_ptr, it->second.size_bytes);
          hipFree(it->second.gpu_ptr);
          io_cache->entries.erase(it);
        }
      }
    }

    // Prefill vs decode detection: decode always has at least one
    // rank>=2 non-KV input with last_dim == 1 (input_ids / position_ids
    // carry a single new token), while prefill's corresponding inputs
    // carry a chunk so min last_dim > 1. Skip rank-0 / rank-1 inputs
    // (scalars / step-length) since they do not distinguish, and skip
    // rank-4 inputs (KV slabs are shaped to max_seq in both modes).
    // Default to prefill on ambiguity -- prefill semantics are strictly
    // safer (no skip), just slower.
    int64_t min_last_dim = -1;
    for (size_t i = 0; i < inputs->count; ++i) {
      const tensor_t &t = inputs->data[i];
      if (!t.data || !t.shape || t.rank < 2 || t.rank == 4)
        continue;
      int64_t last = t.shape[t.rank - 1];
      if (min_last_dim < 0 || last < min_last_dim)
        min_last_dim = last;
    }
    io_cache->is_prefill = (min_last_dim != 1);

    // Steady-decode detection: true iff this call is decode AND every
    // rank-4 input of this call is already in the cache (post-GC). In
    // that state the session is running its own decode loop and
    // finalize_output may skip D2H for rank-4 cache_claim outputs.
    // False on any MISS, on prefill (every call, across all turns and
    // chunks), and on the first decode call of each new generator.
    bool all_hit = !io_cache->current_inputs.empty();
    for (const void *host_ptr : io_cache->current_inputs) {
      if (io_cache->entries.find(host_ptr) == io_cache->entries.end()) {
        all_hit = false;
        break;
      }
    }
    io_cache->skip_rank4_d2h = all_hit && !io_cache->is_prefill;
  }

  // Only rank-4 tensors (KV slabs) are cached. Everything else takes the
  // MISS path below unconditionally: input_ids / attention_mask /
  // position_ids change every step, and a stale cache HIT would break
  // correctness.
  //
  // Two independent decisions on a rank-4 HIT:
  //   gpu_from_cache: reuse the cached GPU buffer instead of pool_alloc
  //                   (always true on HIT -- saves ~32 MiB / slab on
  //                   every call and avoids alloc churn).
  //   skip_h2d      : skip the host->device copy (only safe when this
  //                   state owns the GPU data authoritatively, i.e.
  //                   decode steady state). Prefill always re-copies
  //                   because the host buffer may have been mutated
  //                   since the last call (chat-turn rotation, chunk
  //                   handoff, OGA-side KV trimming).
  void *gpu_ptr = nullptr;
  bool gpu_from_cache = false;
  bool skip_h2d = false;
  if (io_cache && tensor->rank == 4) {
    if (const auto *entry =
            io_cache_lookup(io_cache, tensor->data, size_bytes)) {
      gpu_ptr = entry->gpu_ptr;
      gpu_from_cache = true;
      skip_h2d = !io_cache->is_prefill;
      RUNTIME_DEBUG_LOG(
          "[Runtime DEBUG] prepare_input[%zu]: io_cache HIT host=%p gpu=%p "
          "size=%zu (rank=4, H2D %s)\n",
          index, tensor->data, gpu_ptr, size_bytes,
          skip_h2d ? "skipped (decode)" : "forced (prefill)");
    }
  }

  if (!gpu_from_cache) {
    gpu_ptr = pool_alloc(size_bytes);
    if (!gpu_ptr) {
      fprintf(stderr,
              "hipdnn_ep_tensor_prepare_input: failed to allocate %zu bytes\n",
              size_bytes);
      return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
    }
  }

  if (!skip_h2d) {
    // H2D transfer (instrumented; `io_h2d_input` shows up in the I/O table
    // when HIPDNN_EP_OP_PROFILE=1 along with bytes moved and bandwidth).
    HIPDNN_EP_IO_PROFILE_SCOPE(state, "io_h2d_input", size_bytes);
    if (hipMemcpyAsync(gpu_ptr, tensor->data, size_bytes, hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(state->stream)) != hipSuccess) {
      fprintf(stderr, "hipdnn_ep_tensor_prepare_input: H2D transfer failed\n");
      if (!gpu_from_cache) {
        // Only free pool-allocated buffers on error; the cache still
        // owns cached GPU buffers regardless of the H2D outcome.
        HIP_CLEANUP(hipFree(gpu_ptr));
      }
      return HIPDNN_EP_ERR_H2D_TRANSFER_FAILED;
    }
    perf_count_h2d(size_bytes);
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;

  check_gcnarch("AFTER prepare_input");
  return HIPDNN_EP_SUCCESS;
}

// Prepare output tensor: parse, validate, allocate GPU buffer (no H2D)
int hipdnn_ep_tensor_prepare_output(RuntimeState *state, span_t *outputs,
                                    size_t index, size_t expected_rank,
                                    TensorBuffer *out_buffer) {
  // Validate arguments
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!outputs) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null outputs\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!out_buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_prepare_output: null out_buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate index bounds
  if (index >= outputs->count) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: index %zu out of bounds "
            "(count=%zu)\n",
            index, outputs->count);
    return HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS;
  }

  // Extract tensor from span
  tensor_t *tensor = &outputs->data[index];

  // Validate tensor pointers
  if (!tensor->data) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].data is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!tensor->shape && tensor->rank != 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].shape is null\n",
            index);
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Validate rank
  if (tensor->rank != expected_rank) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: rank mismatch (expected %zu, got "
            "%zu)\n",
            expected_rank, tensor->rank);
    return HIPDNN_EP_ERR_RANK_MISMATCH;
  }

  // Read element size from tensor struct (set by EP caller)
  size_t element_size = tensor->element_size;
  if (element_size == 0) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: tensor[%zu].element_size is 0, "
            "defaulting to %zu\n",
            index, kDefaultElementSize);
    element_size = kDefaultElementSize;
  }

  // Calculate buffer size
  size_t size_bytes =
      calculateTensorSize(tensor->shape, tensor->rank, element_size);
  if (size_bytes == 0) {
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  RUNTIME_DEBUG_LOG(
      "[Runtime DEBUG] prepare_output[%zu]: rank=%zu element_size=%zu "
      "size_bytes=%zu\n",
      index, tensor->rank, element_size, size_bytes);

  // PERF: H2D-end / Compute-start marker. The first prepare_output is
  // called after all prepare_input H2D copies have been queued, so this is
  // a clean phase boundary on the stream.
  if (index == 0)
    perf_record_h2d_end(state->stream);

  // Allocate GPU buffer (pool reuses across inferences)
  void *gpu_ptr = pool_alloc(size_bytes);
  if (!gpu_ptr) {
    fprintf(stderr,
            "hipdnn_ep_tensor_prepare_output: failed to allocate %zu bytes\n",
            size_bytes);
    return HIPDNN_EP_ERR_GPU_ALLOC_FAILED;
  }

  // Populate output buffer
  out_buffer->gpu_ptr = gpu_ptr;
  out_buffer->host_ptr = tensor->data;
  out_buffer->shape_ptr = tensor->shape;
  out_buffer->rank = tensor->rank;
  out_buffer->size_bytes = size_bytes;
  out_buffer->is_pooled = false;

  return HIPDNN_EP_SUCCESS;
}

// Finalize output tensor: D2H transfer, sync, release buffer
int hipdnn_ep_tensor_finalize_output(RuntimeState *state,
                                     TensorBuffer *buffer) {
  if (!state) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_finalize_output: null buffer\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  int result = HIPDNN_EP_SUCCESS;

  // I/O cache decision:
  //   cache_claim: install in the state's cache iff caching is enabled,
  //                this output is rank 4 (KV slab), AND its host_ptr
  //                aliases an input of the same call
  //                (past_present_share_buffer).
  //   D2H skip  : additionally skip D2H iff cache_claim AND the state
  //                is in steady-decode mode (decode call AND every
  //                rank-4 input of this call was a cache HIT post
  //                gen-boundary GC, see skip_rank4_d2h).
  //                Rationale: in steady-decode, the session hands KV
  //                off to itself on the next call via the cache, and
  //                no other session reads the host buffer -- so D2H is
  //                pure waste. Prefill ALWAYS runs D2H (is_prefill
  //                forces skip_rank4_d2h=false) because the next prefill
  //                chunk, the next chat turn, or the decode handoff all
  //                read KV from the host buffer. The first decode call
  //                of each new generator is all-MISS too, which forces
  //                D2H there and keeps the invariant cheaply.
  IoCache *io_cache = io_cache_enabled() ? io_cache_get(state) : nullptr;
  bool cache_claim = io_cache && buffer->rank == 4 &&
                     io_cache->current_inputs.count(buffer->host_ptr) > 0;
  bool skip_d2h = cache_claim && io_cache->skip_rank4_d2h;

  // PERF: D2H-start / Compute-end marker. Idempotent -- only the first
  // finalize of this inference actually queues the event.
  perf_record_d2h_start(state->stream);

  if (!skip_d2h) {
    // D2H transfer (async -- sync happens once after all outputs).
    // Instrumented: shows up as `io_d2h_output` in the profiler I/O
    // table.
    HIPDNN_EP_IO_PROFILE_SCOPE(state, "io_d2h_output", buffer->size_bytes);
    if (hipMemcpyAsync(buffer->host_ptr, buffer->gpu_ptr, buffer->size_bytes,
                       hipMemcpyDeviceToHost,
                       static_cast<hipStream_t>(state->stream)) != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_tensor_finalize_output: D2H transfer failed\n");
      result = HIPDNN_EP_ERR_D2H_TRANSFER_FAILED;
      // Continue to cleanup even on error (best-effort)
    }
    perf_count_d2h(buffer->size_bytes);
  }

  // Re-record d2h_end on the stream right after queueing the real D2H
  // (or right here on the skip path), and sample host_end. Both are
  // last-write-wins, so after the LAST finalize of this inference
  // d2h_end fires immediately after the final D2H copy (or skip-marker)
  // and host_end pins to "right after the EP's last bookkeeping return".
  // This is what makes Wall and d2h_ms accurate for sparse-call models
  // (e.g. the prompt model in OGA's chunked decoder pipeline).
  perf_record_d2h_end(state->stream);
  perf_touch_host_end();

  if (cache_claim) {
    // Hand ownership of the GPU buffer to the cache. The PREVIOUS entry
    // under this host_ptr (if any) was consumed as an input earlier in
    // this call; free_input will pool_release it because its gpu_ptr
    // no longer matches the cache's new entry.
    io_cache_install(io_cache, buffer->host_ptr, buffer->gpu_ptr,
                     buffer->size_bytes);
    RUNTIME_DEBUG_LOG(
        "[Runtime DEBUG] finalize_output: io_cache PIN host=%p gpu=%p "
        "size=%zu (rank=4, D2H %s, cache owns it)\n",
        buffer->host_ptr, buffer->gpu_ptr, buffer->size_bytes,
        skip_d2h ? "SKIPPED (steady decode)"
                 : (io_cache->is_prefill ? "ran (prefill)"
                                         : "ran (decode first call)"));
    buffer->gpu_ptr = nullptr; // cache now owns this GPU buffer
    return result;
  }

  // Non-cached output (rank != 4 or host_ptr not in alive set): return
  // the buffer to the pool.
  pool_release(buffer->gpu_ptr, buffer->size_bytes);
  buffer->gpu_ptr = nullptr;

  return result;
}

// Synchronize GPU stream once (called after all finalize_output calls).
//
// In the current generated inference_compute, this function is not driven
// on the steady-state path -- the next inference's prepare_input(0)
// implicitly orders work via the shared stream. PERF deliberately does NOT
// roll up here, because per-inference rollup would require per-inference
// hipEventSynchronize and that would force CPU/GPU serialization.
int hipdnn_ep_stream_sync(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "hipdnn_ep_stream_sync: null state\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }
  if (hipStreamSynchronize(static_cast<hipStream_t>(state->stream)) !=
      hipSuccess) {
    fprintf(stderr, "hipdnn_ep_stream_sync: stream sync failed\n");
    return HIPDNN_EP_ERR_STREAM_SYNC_FAILED;
  }
  return HIPDNN_EP_SUCCESS;
}

// Release input tensor buffer (no D2H transfer needed)
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer) {
  if (!buffer) {
    fprintf(stderr, "hipdnn_ep_tensor_free_input: null buffer\n");
    return;
  }

  // If THIS state's I/O cache currently owns this (host_ptr, gpu_ptr)
  // pair, leave it alone -- it will serve as a future prepare_input HIT
  // within this same state. The cache, not the per-call TensorBuffer,
  // owns the buffer's lifetime; it is released either when a later
  // finalize_output overwrites the entry (the displaced gpu_ptr is
  // pool_released here on the *next* free_input, since by then it no
  // longer matches the cache) or when the state is destroyed (handled
  // in hipdnn_ep_io_cache_destroy).
  //
  // Only rank-4 buffers can be cached, so this check is skipped for
  // scalar/rank-2/rank-3 inputs to avoid a pointless hashmap lookup.
  if (state && state->io_cache && io_cache_enabled() && buffer->rank == 4 &&
      buffer->host_ptr && buffer->gpu_ptr) {
    auto *io_cache = static_cast<IoCache *>(state->io_cache);
    if (io_cache_owns(io_cache, buffer->host_ptr, buffer->gpu_ptr)) {
      buffer->gpu_ptr = nullptr;
      return;
    }
  }

  // Return buffer to pool
  pool_release(buffer->gpu_ptr, buffer->size_bytes);
  buffer->gpu_ptr = nullptr;
}

//===----------------------------------------------------------------------===//
// TensorBuffer Field Accessors (Opaque Pattern)
//===----------------------------------------------------------------------===//

void *hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_gpu_ptr: null buffer");
  return buffer ? buffer->gpu_ptr : nullptr;
}

void *hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_host_ptr: null buffer");
  return buffer ? buffer->host_ptr : nullptr;
}

int64_t *hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer *buffer) {
  assert(buffer && "get_shape_ptr: null buffer");
  return buffer ? buffer->shape_ptr : nullptr;
}

size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer *buffer) {
  assert(buffer && "get_rank: null buffer");
  return buffer ? buffer->rank : 0;
}

size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer *buffer) {
  assert(buffer && "get_size_bytes: null buffer");
  return buffer ? buffer->size_bytes : 0;
}
