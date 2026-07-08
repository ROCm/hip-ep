/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "op_profile.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct OpProfileState {
  struct ShapeEntry {
    std::string shape;
    double gpuMs = 0;
    double cpuMs = 0;
    int64_t count = 0;
    int64_t bytes = 0; // total data footprint across calls (0 = unknown)
  };
  struct OpEntry {
    std::string name;
    bool hasGpu = false;
    std::map<std::string, ShapeEntry> shapes;
    double totalGpuMs = 0;
    double totalCpuMs = 0;
    int64_t totalCount = 0;
    int64_t totalBytes = 0;
  };
  struct PendingEvent {
    std::string name;
    std::string shape;
    int eventIndex; // pair index (classic) OR marker index (timeline)
    double cpuMs;
    int64_t bytes;
    bool timeline;    // true => eventIndex is a marker index into markerPool
    double cpuStartUs; // op start, us since inference CPU epoch (chrome trace)
  };

  // Chrome-trace timeline point (retained only when HIPDNN_EP_TRACE_FILE set).
  struct TimelineEvent {
    std::string name, shape;
    double tsUs, durUs; // GPU track (epoch-anchored)
    double cpuTsUs, cpuDurUs; // CPU track (wrapper wall time)
    int64_t bytes;
  };
  // H2D / Compute / D2H phase span for the pipeline track (own tids).
  struct IoSpan {
    std::string name;
    int tid;
    double tsUs, durUs;
    int64_t bytes;
  };

  // Pre-allocated event pool: each pair is (start, stop).
  // Grows on demand but never shrinks — avoids hipEventCreate/Destroy per op.
  struct EventPair {
    hipEvent_t start;
    hipEvent_t stop;
  };
  std::vector<EventPair> eventPool;
  int nextEventIndex = 0;

  // Timeline (low-distortion) mode: a single fenceless marker event per op plus
  // one per-inference epoch event. Per-op GPU time is derived by differencing
  // consecutive markers against the epoch on the (in-order) stream, so we pay
  // one fenceless record per op instead of a fenced start+stop pair.
  hipEvent_t epoch = nullptr;
  bool epochRecorded = false;
  std::vector<hipEvent_t> markerPool;
  int nextMarker = 0;

  std::map<std::string, OpEntry> profile;
  std::vector<PendingEvent> pending;
  bool active = false;

  // Chrome trace (HIPDNN_EP_TRACE_FILE): CPU epoch for this inference, all
  // retained timeline points across inferences, and the running wall offset that
  // lays consecutive inferences end-to-end on one timeline.
  double gpuEpochAbsUs = 0.0; // ABSOLUTE us (since the process-wide trace epoch)
                              // at which this inference's GPU epoch was recorded
  std::vector<TimelineEvent> timeline;
  std::vector<IoSpan> ioSpans;
  int sessionId = 0;          // distinguishes concurrent EP sessions (own file)
};

// One process-wide absolute time base for the chrome trace: every EP session
// (embedding, decoder, ...) and every inference share this axis, so traces align
// and merge without per-track/per-inference epoch guessing. Real gaps between
// inferences (idle) show up naturally because absolute time keeps advancing.
static std::chrono::steady_clock::time_point g_traceEpoch;
static bool g_traceEpochSet = false;
static int g_nextSessionId = 0;

OpProfileState *op_profile_create() {
  auto *ps = new OpProfileState;
  ps->sessionId = g_nextSessionId++;
  return ps;
}

void op_profile_destroy(OpProfileState *ps) {
  if (!ps)
    return;
  for (auto &ep : ps->eventPool) {
    hipEventDestroy(ep.start);
    hipEventDestroy(ep.stop);
  }
  for (auto &m : ps->markerPool)
    hipEventDestroy(m);
  if (ps->epoch)
    hipEventDestroy(ps->epoch);
  delete ps;
}

void op_profile_reset(OpProfileState *ps) {
  if (!ps)
    return;
  ps->profile.clear();
  ps->pending.clear();
  ps->nextEventIndex = 0;
  ps->nextMarker = 0;
  ps->epochRecorded = false; // re-anchor the timeline at the next op
  ps->active = true;
  ps->gpuEpochAbsUs = 0.0;
  if (hipdnn_ep_trace_enabled() && !g_traceEpochSet) {
    g_traceEpoch = std::chrono::steady_clock::now();
    g_traceEpochSet = true;
  }
}

double op_profile_cpu_now_us(OpProfileState *ps) {
  if (!ps || !hipdnn_ep_trace_enabled() || !g_traceEpochSet)
    return 0.0;
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - g_traceEpoch)
      .count();
}

bool op_profile_is_active(OpProfileState *ps) { return ps && ps->active; }

// One-time diagnostic sink for HIP-event failures on the profiling path.
// Prints the first few failures with the hipError string + phase so we can
// pinpoint the "invalid resource handle" root cause instead of guessing.
static int g_opProfEventErrs = 0;
void op_profile_note_event_error(const char *phase, int err) {
  if (g_opProfEventErrs < 8) {
    ++g_opProfEventErrs;
    fprintf(stderr, "[PERF-DIAG] hipEvent %s failed: %s (%d)\n", phase,
            hipGetErrorString(static_cast<hipError_t>(err)), err);
  }
}

// Whether to create profiler events with hipEventDisableSystemFence. Defaults
// to OFF (plain hipEventCreate) -- the disable-fence flag is the prime suspect
// for the profiling-path "invalid resource handle". Opt back in with
// HIPDNN_EP_PERF_FENCE_EVENTS=1 to A/B it.
static bool op_profile_use_disable_fence() {
  static bool v = [] {
    const char *e = std::getenv("HIPDNN_EP_PERF_FENCE_EVENTS");
    return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
  }();
  return v;
}

int op_profile_acquire_event_pair(OpProfileState *ps) {
  int idx = ps->nextEventIndex++;
  if (idx >= (int)ps->eventPool.size()) {
    OpProfileState::EventPair ep{};
    // Canonical rationale for hipEventDisableSystemFence across the profiler
    // (other call sites reference this comment): hipEventRecord issues a
    // system-scope acquire/release fence by default, which is wasted work for
    // events whose elapsed time we read only after a hipStreamSynchronize.
    // Disabling it preserves elapsed-time accuracy and GPU completion ordering
    // while removing a per-record fence that, at hundreds of records per
    // inference, otherwise dominated the per-op table.
    //
    // Guardrail: do NOT set this flag on events observed without a following
    // stream/device sync (cross-stream coordination, CPU polling via
    // hipEventQuery, multi-device visibility) -- those rely on the default
    // flush semantics. No profiler event falls into that category.
    //
    // Pool slots are created once and reused across inferences, so creation
    // cost is paid during warmup and every later record is free.
    unsigned flags =
        op_profile_use_disable_fence() ? hipEventDisableSystemFence : hipEventDefault;
    hipError_t e1 = hipEventCreateWithFlags(&ep.start, flags);
    hipError_t e2 = hipEventCreateWithFlags(&ep.stop, flags);
    if (e1 != hipSuccess || e2 != hipSuccess) {
      op_profile_note_event_error("create", e1 != hipSuccess ? e1 : e2);
      // Fall back to plain default events.
      if (e1 == hipSuccess) hipEventDestroy(ep.start);
      if (e2 == hipSuccess) hipEventDestroy(ep.stop);
      hipEventCreate(&ep.start);
      hipEventCreate(&ep.stop);
    }
    ps->eventPool.push_back(ep);
  }
  return idx;
}

hipEvent_t op_profile_get_start_event(OpProfileState *ps, int index) {
  return ps->eventPool[index].start;
}

hipEvent_t op_profile_get_stop_event(OpProfileState *ps, int index) {
  return ps->eventPool[index].stop;
}

// Timeline mode: markers are fenceless (hipEventDisableSystemFence). Their
// elapsed time is read only after the per-inference stream sync, so the default
// system fence is pure overhead here -- and dropping it is the whole point of
// this low-distortion path. See the fence rationale in op_profile_acquire_event_pair.
static hipEvent_t op_profile_make_marker() {
  hipEvent_t e = nullptr;
  hipError_t err = hipEventCreateWithFlags(&e, hipEventDisableSystemFence);
  if (err != hipSuccess || !e) {
    op_profile_note_event_error("create_marker", err);
    hipEventCreate(&e); // fall back to a plain event
  }
  return e;
}

void op_profile_ensure_epoch(OpProfileState *ps, hipStream_t stream) {
  if (!ps || ps->epochRecorded)
    return;
  if (!ps->epoch)
    ps->epoch = op_profile_make_marker();
  hipError_t e = hipEventRecord(ps->epoch, stream);
  if (e != hipSuccess)
    op_profile_note_event_error("record_epoch", e);
  (void)hipGetLastError(); // consume hipEventRecord sticky artifact (see scope)
  ps->epochRecorded = true;
  // Absolute time (on the shared trace axis) at which the GPU epoch was
  // recorded. Adding this to each op's GPU-elapsed offset places the GPU track
  // on the same absolute clock as the CPU track -> aligned, no epoch guessing.
  ps->gpuEpochAbsUs = op_profile_cpu_now_us(ps);
}

int op_profile_acquire_marker(OpProfileState *ps) {
  int idx = ps->nextMarker++;
  if (idx >= (int)ps->markerPool.size())
    ps->markerPool.push_back(op_profile_make_marker());
  return idx;
}

hipEvent_t op_profile_get_marker_event(OpProfileState *ps, int index) {
  return ps->markerPool[index];
}

void op_profile_add_pending(OpProfileState *ps, const std::string &name,
                            const std::string &shape, int eventIndex,
                            double cpuMs, int64_t bytes, bool timeline,
                            double cpuStartUs) {
  if (!ps)
    return;
  ps->pending.push_back(
      {name, shape, eventIndex, cpuMs, bytes, timeline, cpuStartUs});
}

static void write_chrome_trace(OpProfileState *ps);

void op_profile_add_io_spans(OpProfileState *ps, double h2dMs, int64_t h2dBytes,
                             double computeMs, double d2hMs, int64_t d2hBytes) {
  // Called from the tensor runtime's per-inference flush for the inference that
  // just resolved. Placed on dedicated pipeline tracks (H2D/Compute/D2H) within
  // that inference's window (from its GPU epoch). Placement is approximate -- the
  // phase events use a different epoch than the per-op GPU track -- so treat
  // these as the pipeline breakdown, not perfectly aligned to the op spans.
  if (!ps || !hipdnn_ep_trace_enabled())
    return;
  double t = ps->gpuEpochAbsUs; // absolute start of the just-resolved inference
  ps->ioSpans.push_back({"H2D", 2, t, h2dMs * 1000.0, h2dBytes});
  t += h2dMs * 1000.0;
  ps->ioSpans.push_back({"Compute", 4, t, computeMs * 1000.0, 0});
  t += computeMs * 1000.0;
  ps->ioSpans.push_back({"D2H", 3, t, d2hMs * 1000.0, d2hBytes});
  write_chrome_trace(ps);
}

double op_profile_roofline_gbps() {
  static double gbps = [] {
    if (const char *e = std::getenv("HIPDNN_EP_ROOFLINE_GBPS")) {
      double v = atof(e);
      if (v > 0)
        return v;
    }
    return 256.0; // Strix Halo LPDDR5X default
  }();
  return gbps;
}

void op_profile_add_cpu(OpProfileState *ps, const std::string &name,
                        double cpuMs) {
  if (!ps)
    return;
  auto &op = ps->profile[name];
  if (op.name.empty())
    op.name = name;
  auto &sh = op.shapes[""];
  sh.cpuMs += cpuMs;
  sh.count++;
  op.totalCpuMs += cpuMs;
  op.totalCount++;
}

// Chrome trace writer (chrome://tracing / Perfetto). Emits one GPU track
// (epoch-anchored, faithful in timeline mode) and one CPU track (wrapper wall
// time) with shape/bytes/GB-s args. Rewrites the whole file each inference so
// the trace stays valid mid-run; events from all inferences sit at their real
// absolute times on the shared axis. Gated by HIPDNN_EP_TRACE_FILE.
static void write_chrome_trace(OpProfileState *ps) {
  const std::string &base = hipdnn_ep_trace_path();
  if (base.empty())
    return;
  // Per-session file: insert ".sN" before the extension so concurrent EP
  // sessions (embedding, decoder, ...) don't clobber one shared file. They all
  // share the absolute time axis, so merge_traces.py aligns them exactly.
  size_t dot = base.find_last_of('.');
  std::string path = (dot == std::string::npos)
                         ? base + ".s" + std::to_string(ps->sessionId)
                         : base.substr(0, dot) + ".s" +
                               std::to_string(ps->sessionId) + base.substr(dot);
  FILE *f = fopen(path.c_str(), "w");
  if (!f)
    return;
  fputs("{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n", f);
  fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,"
             "\"args\":{\"name\":\"hipdnn-ep session %d\"}},\n", ps->sessionId);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,"
        "\"args\":{\"name\":\"CPU (wrapper)\"}},\n", f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":1,"
        "\"args\":{\"name\":\"GPU (stream)\"}},\n", f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":2,"
        "\"args\":{\"name\":\"H2D\"}},\n", f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":3,"
        "\"args\":{\"name\":\"D2H\"}},\n", f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":4,"
        "\"args\":{\"name\":\"Compute (phase)\"}}", f);
  const double peak = op_profile_roofline_gbps();
  for (auto &s : ps->ioSpans) {
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"io\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"bytes\":%lld}}",
            s.name.c_str(), s.tid, s.tsUs, s.durUs, (long long)s.bytes);
  }
  for (auto &e : ps->timeline) {
    double gbps = (e.bytes > 0 && e.durUs > 0)
                      ? (double)e.bytes / 1e3 / e.durUs
                      : 0.0; // bytes / us = GB/s * 1e-... : bytes/(us)/1e3
    // GPU span
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"gpu\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":1,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"shape\":\"%s\","
            "\"bytes\":%lld,\"GB/s\":%.0f,\"%%peak\":%.0f}}",
            e.name.c_str(), e.tsUs, e.durUs, e.shape.c_str(),
            (long long)e.bytes, gbps, peak > 0 ? 100.0 * gbps / peak : 0.0);
    // CPU span (host-side wrapper time: launch overhead + setup)
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"cpu\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":0,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"shape\":\"%s\"}}",
            e.name.c_str(), e.cpuTsUs, e.cpuDurUs, e.shape.c_str());
  }
  fputs("\n]}\n", f);
  fclose(f);
}

void op_profile_resolve_and_print(OpProfileState *ps) {
  if (!ps)
    return;

  // Timeline mode: each op's GPU time is the gap between its marker and the
  // previous marker (or the epoch for the first op) on the in-order stream.
  // pending is in enqueue order, so cumulative-from-epoch differencing yields
  // per-op durations without any per-op start event.
  const bool traceOn = hipdnn_ep_trace_enabled();
  double tlPrevMs = 0.0;
  for (auto &ev : ps->pending) {
    float gpuMs = 0.0f;
    double gpuStartMs = -1.0; // epoch-relative start (timeline mode only)
    if (ev.timeline) {
      if (ps->epoch) {
        float cumMs = 0.0f;
        hipError_t elErr = hipEventElapsedTime(
            &cumMs, ps->epoch, ps->markerPool[ev.eventIndex]);
        if (elErr != hipSuccess) {
          op_profile_note_event_error("elapsed_marker", elErr);
          cumMs = (float)tlPrevMs;
        }
        gpuMs = (float)(cumMs - tlPrevMs);
        if (gpuMs < 0.0f)
          gpuMs = 0.0f; // guard against event resolution jitter
        gpuStartMs = tlPrevMs; // op ran from prev marker to this one
        tlPrevMs = cumMs;
      }
    } else {
      hipError_t elErr = hipEventElapsedTime(
          &gpuMs, ps->eventPool[ev.eventIndex].start,
          ps->eventPool[ev.eventIndex].stop);
      if (elErr != hipSuccess) {
        op_profile_note_event_error("elapsed", elErr);
        gpuMs = 0.0f;
      }
      // Epoch-anchored start for classic mode (faithful chrome-trace placement).
      if (traceOn && ps->epoch) {
        float st = 0.0f;
        if (hipEventElapsedTime(&st, ps->epoch,
                                ps->eventPool[ev.eventIndex].start) ==
            hipSuccess)
          gpuStartMs = st;
      }
    }
    if (traceOn) {
      // Absolute timestamps on the shared axis. GPU start = inference's GPU
      // epoch (absolute) + this op's GPU-elapsed offset; CPU start = absolute
      // wrapper-entry time. Both already share the process trace epoch, so no
      // per-track/per-inference re-basing is needed.
      double gTsUs = (gpuStartMs >= 0.0)
                         ? ps->gpuEpochAbsUs + gpuStartMs * 1000.0
                         : ev.cpuStartUs;
      ps->timeline.push_back({ev.name, ev.shape, gTsUs, gpuMs * 1000.0,
                              ev.cpuStartUs, ev.cpuMs * 1000.0, ev.bytes});
    }
    auto &op = ps->profile[ev.name];
    if (op.name.empty())
      op.name = ev.name;
    op.hasGpu = true;
    auto &sh = op.shapes[ev.shape];
    if (sh.shape.empty() && !ev.shape.empty())
      sh.shape = ev.shape;
    sh.gpuMs += gpuMs;
    sh.cpuMs += ev.cpuMs;
    sh.count++;
    sh.bytes += ev.bytes;
    op.totalGpuMs += gpuMs;
    op.totalCpuMs += ev.cpuMs;
    op.totalCount++;
    op.totalBytes += ev.bytes;
  }
  ps->pending.clear();

  if (traceOn)
    write_chrome_trace(ps); // absolute-axis; per-session file (own sessionId)

  // Consume any sticky HIP error introduced by the profiler's own event calls
  // (e.g. a hipEventElapsedTime on a not-fully-ready event). Otherwise the next
  // Compute's first kernel wrapper that calls hipGetLastError() (hip_reduce_sum,
  // the generic elementwise path) misattributes it as its own launch failure --
  // the root cause of the "invalid resource handle" seen only under profiling.
  (void)hipGetLastError();

  if (ps->profile.empty())
    return;

  struct OpRow {
    std::string name;
    bool hasGpu;
    double totalGpuMs;
    double totalCpuMs;
    int64_t totalCount;
    int64_t totalBytes;
    std::vector<OpProfileState::ShapeEntry> shapes;
  };

  std::vector<OpRow> rows;
  int maxNameLen = 22;
  for (auto &[_, op] : ps->profile) {
    OpRow row;
    row.name = op.name;
    row.hasGpu = op.hasGpu;
    row.totalGpuMs = op.totalGpuMs;
    row.totalCpuMs = op.totalCpuMs;
    row.totalCount = op.totalCount;
    row.totalBytes = op.totalBytes;
    if ((int)row.name.size() > maxNameLen)
      maxNameLen = (int)row.name.size();
    for (auto &[_, sh] : op.shapes) {
      row.shapes.push_back(sh);
      int shapeNameLen = sh.shape.empty() ? 0 : (int)sh.shape.size() + 2;
      if (shapeNameLen > maxNameLen)
        maxNameLen = shapeNameLen;
    }
    std::sort(
        row.shapes.begin(), row.shapes.end(),
        [](const OpProfileState::ShapeEntry &a,
           const OpProfileState::ShapeEntry &b) { return a.gpuMs > b.gpuMs; });
    rows.push_back(std::move(row));
  }

  std::sort(rows.begin(), rows.end(), [](const OpRow &a, const OpRow &b) {
    if (a.hasGpu != b.hasGpu)
      return a.hasGpu;
    if (a.hasGpu)
      return a.totalGpuMs > b.totalGpuMs;
    return a.totalCpuMs > b.totalCpuMs;
  });

  double grandTotalGpuMs = 0, grandTotalCpuMs = 0;
  for (auto &r : rows) {
    if (r.hasGpu)
      grandTotalGpuMs += r.totalGpuMs;
    grandTotalCpuMs += r.totalCpuMs;
  }

  const double peakGbps = op_profile_roofline_gbps();
  // Achieved bandwidth (GB/s) and % of the memory roofline for a (bytes, ms)
  // pair; empty strings when bytes are unknown (op not byte-instrumented).
  auto fmtBw = [peakGbps](int64_t bytes, double ms, char *mbBuf, char *gbBuf,
                          char *pkBuf, size_t n) {
    if (bytes > 0 && ms > 0) {
      double gbps = (double)bytes / 1e9 / (ms / 1000.0);
      snprintf(mbBuf, n, "%.1f", (double)bytes / 1e6);
      snprintf(gbBuf, n, "%.0f", gbps);
      snprintf(pkBuf, n, "%.0f%%", 100.0 * gbps / peakGbps);
    } else {
      snprintf(mbBuf, n, "%s", "-");
      snprintf(gbBuf, n, "%s", "-");
      snprintf(pkBuf, n, "%s", "-");
    }
  };
  char mbBuf[24], gbBuf[24], pkBuf[24];

  int lineWidth = maxNameLen + 2 + 5 + 1 + 9 + 1 + 9 + 1 + 6 + 1 + 9 + 1 + 8 + 1 + 6;
  fprintf(stderr, "\n[PERF] ");
  for (int i = 0; i < lineWidth; ++i)
    fputc('=', stderr);
  fputc('\n', stderr);
  fprintf(stderr, "[PERF]  %-*s %5s %9s %9s %6s %9s %8s %6s   (roofline %.0f GB/s)\n",
          maxNameLen, "", "calls", "gpu (ms)", "cpu (ms)", "gpu %", "MB",
          "GB/s", "%pk", peakGbps);

  for (auto &r : rows) {
    if (r.hasGpu) {
      double pct =
          grandTotalGpuMs > 0 ? r.totalGpuMs / grandTotalGpuMs * 100 : 0;
      fmtBw(r.totalBytes, r.totalGpuMs, mbBuf, gbBuf, pkBuf, sizeof(mbBuf));
      fprintf(stderr, "[PERF]  %-*s %5lld %9.1f %9.1f %5.1f%% %9s %8s %6s\n",
              maxNameLen, r.name.c_str(), (long long)r.totalCount, r.totalGpuMs,
              r.totalCpuMs, pct, mbBuf, gbBuf, pkBuf);
    } else {
      fprintf(stderr, "[PERF]  %-*s %5lld %9s %9.1f %6s\n", maxNameLen,
              r.name.c_str(), (long long)r.totalCount, "n/a", r.totalCpuMs,
              "n/a");
    }
    bool hasShapes = r.shapes.size() > 1 ||
                     (r.shapes.size() == 1 && !r.shapes[0].shape.empty());
    if (hasShapes) {
      for (auto &sh : r.shapes) {
        if (sh.shape.empty())
          continue;
        double pct = grandTotalGpuMs > 0 ? sh.gpuMs / grandTotalGpuMs * 100 : 0;
        fmtBw(sh.bytes, sh.gpuMs, mbBuf, gbBuf, pkBuf, sizeof(mbBuf));
        fprintf(stderr, "[PERF]    %-*s %5lld %9.1f %9.1f %5.1f%% %9s %8s %6s\n",
                maxNameLen - 2, sh.shape.c_str(), (long long)sh.count, sh.gpuMs,
                sh.cpuMs, pct, mbBuf, gbBuf, pkBuf);
      }
    }
  }
  fprintf(stderr, "[PERF]  %-*s %5s %9.1f %9.1f\n", maxNameLen, "TOTAL", "",
          grandTotalGpuMs, grandTotalCpuMs);
  fprintf(stderr, "[PERF] ");
  for (int i = 0; i < lineWidth; ++i)
    fputc('=', stderr);
  fprintf(stderr, "\n\n");
  ps->profile.clear();
}
