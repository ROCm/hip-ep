/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "operator_profile.h"
#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_ep {
namespace {

// Per-operator running statistics. Used for both compute and I/O entries.
// `total_bytes`/`is_io` are zero/false for compute ops and populated for I/O.
struct OpStats {
  size_t count = 0;
  double total_ms = 0.0;
  double min_ms = 0.0; // initialized lazily on first sample
  double max_ms = 0.0;
  size_t total_bytes = 0;
  bool is_io = false;
};

// Wrap a {warmup, steady, count} triple so the rest of the file can keep
// passing one logical "aggregate" around.
struct OpAggregate {
  std::unordered_map<std::string, OpStats> warmup;
  std::unordered_map<std::string, OpStats> steady;
  // Inferences that contributed to each bucket. Used for the report header
  // ("over N warmup + M steady inference(s)").
  unsigned warmup_count = 0;
  unsigned steady_count = 0;

  bool empty() const { return warmup.empty() && steady.empty(); }
  void clear() {
    warmup.clear();
    steady.clear();
    warmup_count = 0;
    steady_count = 0;
  }
};

// One pending start/end event pair recorded by op_profile_scope_*.
struct PendingEvent {
  const char *name; // points to static storage (typically __func__)
  hipEvent_t start;
  hipEvent_t end;
  size_t bytes; // 0 for compute ops, transfer size for I/O ops
  bool is_io;
};

// HIP event recycling pool. hipEventCreate is not free, so we keep a list of
// reusable events and only allocate when the pool is empty.
struct EventPool {
  std::vector<hipEvent_t> free_list;

  hipEvent_t acquire() {
    if (!free_list.empty()) {
      hipEvent_t e = free_list.back();
      free_list.pop_back();
      return e;
    }
    hipEvent_t e = nullptr;
    if (hipEventCreate(&e) != hipSuccess)
      return nullptr;
    return e;
  }

  void release(hipEvent_t e) {
    if (e)
      free_list.push_back(e);
  }

  void destroy_all() {
    for (auto e : free_list) {
      if (e)
        (void)hipEventDestroy(e);
    }
    free_list.clear();
  }
};

// All globals are accessed from a single host thread (the runtime is not
// thread-safe per RuntimeState), so no locking is required.
static std::vector<PendingEvent> g_pending;
// Per-session aggregate (reset on each op_profile_print_summary_and_reset).
// In OGA's chunked decoder pipeline this corresponds to the prompt model and
// the decoder model getting their own [OP_PROFILE] block, mirroring [PERF].
static OpAggregate g_aggregate;
static EventPool g_event_pool;
// Index of the inference whose events are currently in g_pending. Used to
// route those events to the warmup or steady bucket when they're flushed.
// Resets to 0 when a session aggregate is printed so the next session's
// warmup_count first inferences land in its warmup bucket.
static unsigned g_inference_idx = 0;
// Counter for the [OP_PROFILE] block heading. Mirrors g_perf_session_index
// in hipdnn_ep_runtime_tensor.cpp so the user can pair op-profile and perf
// blocks one-to-one.
static unsigned g_session_index = 0;

// Stash the most recently recorded "end" event so flush_inference can
// hipEventSynchronize on a single event to ensure ALL prior events on the
// stream have completed (events on the same stream complete in record order).
static hipEvent_t g_last_end_event = nullptr;

void update_stats(OpStats &s, double ms, size_t bytes, bool is_io) {
  // hipEventElapsedTime can return small negative values (~few us) when the
  // start/end events resolve at the same GPU clock tick. Clamp to 0 so the
  // running min/total cannot drift below zero.
  if (ms < 0.0)
    ms = 0.0;
  if (s.count == 0) {
    s.min_ms = ms;
    s.max_ms = ms;
  } else {
    if (ms < s.min_ms)
      s.min_ms = ms;
    if (ms > s.max_ms)
      s.max_ms = ms;
  }
  s.count++;
  s.total_ms += ms;
  s.total_bytes += bytes;
  if (is_io)
    s.is_io = true;
}

// Format a byte count as a short human-readable string (e.g. "1.45 GiB").
// Uses a fixed-size buffer so callers can embed it in fprintf without heap.
void format_bytes(char *out, size_t out_size, size_t bytes) {
  const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(bytes);
  unsigned u = 0;
  while (v >= 1024.0 && u + 1 < sizeof(units) / sizeof(units[0])) {
    v /= 1024.0;
    ++u;
  }
  if (u == 0)
    std::snprintf(out, out_size, "%zu %s", bytes, units[u]);
  else
    std::snprintf(out, out_size, "%.2f %s", v, units[u]);
}

// Print one bucket's compute + I/O tables. Used by print_summary() to emit
// the warmup section and the steady-state section back-to-back.
void print_one_bucket(FILE *out,
                      const std::unordered_map<std::string, OpStats> &m) {
  if (m.empty())
    return;

  // Partition into compute operators and I/O entries so each can be reported
  // with the columns most useful to it (I/O gets a bandwidth column).
  std::vector<std::pair<std::string, OpStats>> ops, ios;
  ops.reserve(m.size());
  ios.reserve(m.size());
  for (const auto &kv : m) {
    if (kv.second.is_io)
      ios.push_back(kv);
    else
      ops.push_back(kv);
  }

  auto sort_by_total = [](std::vector<std::pair<std::string, OpStats>> &v) {
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string, OpStats> &a,
                 const std::pair<std::string, OpStats> &b) {
                return a.second.total_ms > b.second.total_ms;
              });
  };
  sort_by_total(ops);
  sort_by_total(ios);

  // ---- Operator (compute) table ----
  if (!ops.empty()) {
    double grand_total_ms = 0.0;
    size_t grand_count = 0;
    for (const auto &kv : ops) {
      grand_total_ms += kv.second.total_ms;
      grand_count += kv.second.count;
    }
    fprintf(out, "  %-32s %8s %10s %10s %10s %10s %7s\n", "operator", "count",
            "total_ms", "avg_ms", "min_ms", "max_ms", "%total");
    for (const auto &kv : ops) {
      const auto &s = kv.second;
      double avg_ms =
          s.count ? (s.total_ms / static_cast<double>(s.count)) : 0.0;
      double pct = grand_total_ms > 0.0
                       ? (s.total_ms / grand_total_ms * 100.0)
                       : 0.0;
      fprintf(out, "  %-32s %8zu %10.3f %10.3f %10.3f %10.3f %6.1f%%\n",
              kv.first.c_str(), s.count, s.total_ms, avg_ms, s.min_ms, s.max_ms,
              pct);
    }
    fprintf(out, "  %-32s %8zu %10.3f\n", "TOTAL", grand_count, grand_total_ms);
  }

  // ---- I/O (memcpy) table ----
  if (!ios.empty()) {
    double io_total_ms = 0.0;
    size_t io_total_count = 0;
    size_t io_total_bytes = 0;
    for (const auto &kv : ios) {
      io_total_ms += kv.second.total_ms;
      io_total_count += kv.second.count;
      io_total_bytes += kv.second.total_bytes;
    }
    fprintf(out, "  --- I/O (memcpy) ---\n");
    fprintf(out, "  %-32s %8s %10s %10s %14s %10s %7s\n", "io_path", "count",
            "total_ms", "avg_ms", "total_bytes", "GB/s", "%total");
    for (const auto &kv : ios) {
      const auto &s = kv.second;
      double avg_ms =
          s.count ? (s.total_ms / static_cast<double>(s.count)) : 0.0;
      // Bandwidth: total_bytes / total_seconds, expressed in GB/s (1e9).
      double gbs = (s.total_ms > 0.0)
                       ? (static_cast<double>(s.total_bytes) /
                          (s.total_ms * 1.0e6))
                       : 0.0;
      double pct =
          io_total_ms > 0.0 ? (s.total_ms / io_total_ms * 100.0) : 0.0;
      char bytes_str[24];
      format_bytes(bytes_str, sizeof(bytes_str), s.total_bytes);
      fprintf(out, "  %-32s %8zu %10.3f %10.3f %14s %10.2f %6.1f%%\n",
              kv.first.c_str(), s.count, s.total_ms, avg_ms, bytes_str, gbs,
              pct);
    }
    char bytes_str[24];
    format_bytes(bytes_str, sizeof(bytes_str), io_total_bytes);
    double total_gbs =
        io_total_ms > 0.0
            ? (static_cast<double>(io_total_bytes) / (io_total_ms * 1.0e6))
            : 0.0;
    fprintf(out, "  %-32s %8zu %10.3f %10s %14s %10.2f\n", "TOTAL",
            io_total_count, io_total_ms, "", bytes_str, total_gbs);
  }
}

// Print the full session aggregate -- a warmup block followed by a steady-
// state block, each with its own compute + I/O tables. The session label
// ("session #N") matches the [PERF] session label so warmup / steady
// counts line up across the two reports for the same model.
void print_summary(FILE *out, unsigned session_index,
                   const OpAggregate &agg) {
  if (agg.empty())
    return;
  if (agg.warmup_count > 0 && !agg.warmup.empty()) {
    fprintf(out,
            "[OP_PROFILE] session #%u WARMUP (first %u inference(s), "
            "kernel selection / JIT / autotune):\n",
            session_index, agg.warmup_count);
    print_one_bucket(out, agg.warmup);
  }
  if (agg.steady_count > 0 && !agg.steady.empty()) {
    fprintf(out,
            "[OP_PROFILE] session #%u STEADY-STATE over %u inference(s):\n",
            session_index, agg.steady_count);
    print_one_bucket(out, agg.steady);
  }
}

} // namespace

// Pick the correct bucket (warmup or steady) for the inference currently
// being flushed. The split point matches PERF's: the first
// hipdnn_ep_perf_warmup_count() inferences in this session go to warmup.
static std::unordered_map<std::string, OpStats> &
bucket_for_current_inference() {
  return (g_inference_idx < hipdnn_ep_perf_warmup_count()) ? g_aggregate.warmup
                                                            : g_aggregate.steady;
}

// Process-exit handler: dump whatever aggregate has accumulated since the last
// state_cleanup. Without this, work performed during the benchmark warmup +
// timed iterations never reaches stderr because the OGA model_benchmark holds
// the EP sessions alive until process exit, and DLL unload is not guaranteed
// to drive state_cleanup.
static void op_profile_atexit_dump() {
  if (g_pending.empty() && g_aggregate.empty())
    return;

  // Best-effort flush of leftover events. We deliberately don't call
  // hipEventSynchronize here -- the runtime stream may already be invalid at
  // process exit. Just consume what we have.
  if (!g_pending.empty()) {
    auto &bucket = bucket_for_current_inference();
    for (auto &e : g_pending) {
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, e.start, e.end) == hipSuccess)
        update_stats(bucket[std::string(e.name)], static_cast<double>(ms),
                     e.bytes, e.is_io);
    }
    // Count this incomplete tail as one extra inference on the right side.
    if (g_inference_idx < hipdnn_ep_perf_warmup_count())
      g_aggregate.warmup_count++;
    else
      g_aggregate.steady_count++;
    g_pending.clear();
    g_last_end_event = nullptr;
  }

  if (!g_aggregate.empty()) {
    ++g_session_index;
    print_summary(stderr, g_session_index, g_aggregate);
  }
  g_aggregate.clear();
}

bool op_profile_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_OP_PROFILE");
    bool on = v && v[0] >= '1';
    if (on)
      std::atexit(op_profile_atexit_dump);
    return on;
  }();
  return enabled;
}

// When HIPDNN_EP_OP_PROFILE_EACH=1 is set, print a summary table after every
// inference. Off by default because LLM decode produces hundreds of inferences
// per benchmark run and the per-inference output would drown out other logs;
// the lifetime aggregate dumped at state_cleanup is usually enough.
static bool op_profile_each_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_OP_PROFILE_EACH");
    return v && v[0] >= '1';
  }();
  return enabled;
}

void *op_profile_scope_begin(RuntimeState *state, const char * /*name*/) {
  if (!op_profile_enabled() || !state)
    return nullptr;
  void *raw_stream = hipdnn_ep_state_get_stream(state);
  if (!raw_stream)
    return nullptr;
  hipEvent_t start = g_event_pool.acquire();
  if (!start)
    return nullptr;
  if (hipEventRecord(start, static_cast<hipStream_t>(raw_stream)) !=
      hipSuccess) {
    g_event_pool.release(start);
    return nullptr;
  }
  return start;
}

// Internal helper shared by op and io scope_end to avoid duplicating the
// event-recording boilerplate. `bytes`/`is_io` are stamped onto the
// PendingEvent so the flush stage can route compute and I/O entries to the
// correct table.
static void scope_end_impl(RuntimeState *state, const char *name, void *token,
                           size_t bytes, bool is_io) {
  if (!token)
    return;
  hipEvent_t start = static_cast<hipEvent_t>(token);
  if (!state) {
    g_event_pool.release(start);
    return;
  }
  void *raw_stream = hipdnn_ep_state_get_stream(state);
  if (!raw_stream) {
    g_event_pool.release(start);
    return;
  }
  hipEvent_t end = g_event_pool.acquire();
  if (!end) {
    g_event_pool.release(start);
    return;
  }
  if (hipEventRecord(end, static_cast<hipStream_t>(raw_stream)) !=
      hipSuccess) {
    g_event_pool.release(start);
    g_event_pool.release(end);
    return;
  }

  PendingEvent ev;
  ev.name = name ? name : "<unknown>";
  ev.start = start;
  ev.end = end;
  ev.bytes = bytes;
  ev.is_io = is_io;
  g_pending.push_back(ev);
  g_last_end_event = end;
}

void op_profile_scope_end(RuntimeState *state, const char *name, void *token) {
  scope_end_impl(state, name, token, /*bytes=*/0, /*is_io=*/false);
}

void *op_profile_io_scope_begin(RuntimeState *state, const char *name,
                                size_t /*bytes*/) {
  // Begin-side is identical to the compute scope -- we just record a HIP
  // event. The byte count is stamped at end-time on the PendingEvent so
  // the flush stage can route it to the I/O table.
  return op_profile_scope_begin(state, name);
}

void op_profile_io_scope_end(RuntimeState *state, const char *name, void *token,
                             size_t bytes) {
  scope_end_impl(state, name, token, bytes, /*is_io=*/true);
}

void op_profile_flush_inference() {
  if (!op_profile_enabled())
    return;
  if (g_pending.empty())
    return;

  // Wait until the last event has completed -- this implies all previous
  // events on the same stream have also completed, so all hipEventElapsedTime
  // queries below will return a valid result.
  if (g_last_end_event)
    (void)hipEventSynchronize(g_last_end_event);

  const bool emit_each = op_profile_each_enabled();
  const bool is_warmup = g_inference_idx < hipdnn_ep_perf_warmup_count();
  auto &bucket = is_warmup ? g_aggregate.warmup : g_aggregate.steady;

  std::unordered_map<std::string, OpStats> per_inference;
  for (auto &e : g_pending) {
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, e.start, e.end) == hipSuccess) {
      const std::string key(e.name);
      if (emit_each)
        update_stats(per_inference[key], static_cast<double>(ms), e.bytes,
                     e.is_io);
      update_stats(bucket[key], static_cast<double>(ms), e.bytes, e.is_io);
    }
    g_event_pool.release(e.start);
    g_event_pool.release(e.end);
  }
  g_pending.clear();
  g_last_end_event = nullptr;

  if (is_warmup)
    g_aggregate.warmup_count++;
  else
    g_aggregate.steady_count++;
  g_inference_idx++;

  if (emit_each) {
    char header[64];
    std::snprintf(header, sizeof(header), "[OP_PROFILE] inference #%u%s:",
                  g_inference_idx, is_warmup ? " [warmup]" : "");
    fprintf(stderr, "%s\n", header);
    print_one_bucket(stderr, per_inference);
  }
}

// Drain leftovers, print the per-session aggregate (warmup + steady), and
// reset the session-local counters so the NEXT session's first inferences
// land in its own warmup bucket. Mirrors PERF's per-session behavior.
void op_profile_print_summary_and_reset() {
  // Drain any leftover events so their resources are released even when
  // profiling is being turned off mid-flight.
  if (!g_pending.empty())
    op_profile_flush_inference();

  if (op_profile_enabled() && !g_aggregate.empty()) {
    ++g_session_index;
    print_summary(stderr, g_session_index, g_aggregate);
  }

  g_aggregate.clear();
  g_inference_idx = 0;
  g_event_pool.destroy_all();
}

} // namespace hipdnn_ep
