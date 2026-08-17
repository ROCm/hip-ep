/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The HIP types and entry points below arrive through op_profile.h ->
// runtime_types.h, which resolves to real/ or mock/ depending on the build.
// Including <hip/hip_runtime.h> directly here would be redundant for the real
// runtime and fatal for the mock one, which has no ROCm headers on its include
// path.
#include "op_profile.h"
#include "chrome_trace.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// One-shot RGP capture fence -- see op_profile.h for the rationale. Pure host
// code (no kernel launch): drains the GPU and sleeps so RGP arms on the next
// dispatch. All env reads are latched on first call; when RGP_FENCE is unset
// the very first check returns and the whole thing compiles down to one load.
void rgp_capture_fence(const char *opname) {
  static const std::string target = hipdnn_ep::env_string("RGP_FENCE");
  if (target.empty())
    return;
  static const int skip = [] {
    const std::string s = hipdnn_ep::env_string("RGP_FENCE_SKIP");
    return s.empty() ? 0 : std::atoi(s.c_str());
  }();
  static const int sleep_ms = [] {
    const std::string s = hipdnn_ep::env_string("RGP_FENCE_MS");
    return s.empty() ? 200 : std::atoi(s.c_str());
  }();
  static std::atomic<bool> fired{false};
  static std::atomic<int> matches{0};

  if (fired.load(std::memory_order_acquire))
    return;
  if (!opname || target != opname)
    return;
  // Arm only on the (skip)-th matching instance so callers can pick, e.g., a
  // full-attention layer rather than the first (sliding) one.
  int idx = matches.fetch_add(1, std::memory_order_acq_rel);
  if (idx < skip)
    return;
  bool expected = false;
  if (!fired.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    return;
  // Drain all outstanding GPU work, THEN announce the idle window and hold. The
  // marker is emitted BEFORE the wait so the capture orchestrator has the full
  // window to detect it and trigger RGP; the GPU stays quiescent for the whole
  // wait, so RGP arms cleanly and this op's kernels are the first dispatches
  // after the gap. Busy-wait on the steady clock (no threading headers, no GPU
  // work).
  (void)hipDeviceSynchronize();
  fprintf(stderr, "[RGP_FENCE_ARMED] op=%s instance=%d idling sleep_ms=%d\n",
          opname, idx, sleep_ms);
  fflush(stderr);
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(sleep_ms);
  while (std::chrono::steady_clock::now() < until) { /* spin */
  }
}

struct OpProfileState {
  struct ShapeEntry {
    std::string shape;
    double gpuMs = 0;
    double cpuMs = 0;
    int64_t count = 0;
  };
  struct OpEntry {
    std::string name;
    bool hasGpu = false;
    bool isOuter = false;
    std::map<std::string, ShapeEntry> shapes;
    double totalGpuMs = 0;
    double totalCpuMs = 0;
    int64_t totalCount = 0;
  };
  struct PendingEvent {
    std::string name;
    std::string shape;
    int markerIndex;
    double cpuMs;
    int64_t bytes;
    double cpuStartUs; // op start, absolute us on the shared trace axis
  };

  // Low-distortion timing: a single fenceless marker event per op plus one
  // per-inference epoch event. Per-op GPU time is derived by differencing
  // consecutive markers against the epoch on the (in-order) stream, so we pay
  // one fenceless record per op instead of a fenced start+stop pair.
  hipEvent_t epoch = nullptr;
  bool epochRecorded = false;
  std::vector<hipEvent_t> markerPool;
  int nextMarker = 0;

  std::map<std::string, OpEntry> profile;
  std::vector<PendingEvent> pending;
  bool active = false;

  // Absolute us (on the shared steady_clock trace axis) at which this
  // inference's GPU epoch event was recorded; per-op GPU spans are placed
  // relative to it. Only meaningful when tracing.
  double gpuEpochAbsUs = 0.0;
  // Chrome-trace accumulator + writer (HIPDNN_EP_TRACE_FILE). Cheap to hold
  // when tracing is off (just an empty vector pair + a session tag).
  ChromeTrace trace;
};

OpProfileState *op_profile_create() { return new OpProfileState; }

void op_profile_destroy(OpProfileState *ps) {
  if (!ps)
    return;
  // Write the trace once, on clean shutdown: write() rewrites the whole file,
  // so doing it per inference would be O(N^2), and the JSON is only consumed
  // after the run (postprocessed / merged).
  if (hipdnn_ep_trace_enabled())
    ps->trace.write(hipdnn_ep_trace_path(), op_profile_roofline_gbps());
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
  ps->nextMarker = 0;
  ps->epochRecorded = false; // re-anchor the timeline at the next op
  ps->active = true;
  ps->gpuEpochAbsUs = 0.0;
}

bool op_profile_is_active(OpProfileState *ps) { return ps && ps->active; }

// Markers are fenceless (hipEventDisableSystemFence). Their
// elapsed time is read only after the per-inference stream sync, so the default
// system-scope fence would be pure overhead here -- dropping it is the whole
// point of this low-distortion path.
static hipEvent_t op_profile_make_marker() {
  hipEvent_t e = nullptr;
  hipError_t err = hipEventCreateWithFlags(&e, hipEventDisableSystemFence);
  if (err != hipSuccess || !e)
    hipEventCreate(&e); // fall back to a plain event
  return e;
}

void op_profile_ensure_epoch(OpProfileState *ps, hipStream_t stream) {
  if (!ps || ps->epochRecorded)
    return;
  if (!ps->epoch)
    ps->epoch = op_profile_make_marker();
  (void)hipEventRecord(ps->epoch, stream);
  ps->epochRecorded = true;
  // Absolute time (on the shared trace axis) at which the GPU epoch was
  // recorded. Adding this to each op's GPU-elapsed offset places the GPU track
  // on the same absolute clock as the CPU track -> aligned, no epoch guessing.
  ps->gpuEpochAbsUs =
      op_profile_us_since_epoch(std::chrono::steady_clock::now());
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
                            const std::string &shape, int markerIndex,
                            double cpuMs, int64_t bytes, double cpuStartUs) {
  if (!ps)
    return;
  ps->pending.push_back({name, shape, markerIndex, cpuMs, bytes, cpuStartUs});
}

void op_profile_add_io_spans(OpProfileState *ps, double h2dMs, int64_t h2dBytes,
                             double computeMs, double d2hMs, int64_t d2hBytes) {
  // Called from the tensor runtime's per-inference flush for the inference that
  // just resolved. Placed on dedicated pipeline tracks (H2D/Compute/D2H) within
  // that inference's window (from its GPU epoch). Placement is approximate --
  // the phase events use a different epoch than the per-op GPU track -- so
  // treat these as the pipeline breakdown, not perfectly aligned to the op
  // spans.
  if (!ps || !hipdnn_ep_trace_enabled())
    return;
  ps->trace.addIoSpans(ps->gpuEpochAbsUs, h2dMs, h2dBytes, computeMs, d2hMs,
                       d2hBytes);
}

double op_profile_roofline_gbps() {
  // TODO: query the device's peak memory bandwidth instead of hardcoding.
  return 256.0; // Strix Halo LPDDR5X
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

void op_profile_add_cpu_total(OpProfileState *ps, const std::string &name,
                              double cpuStartUs, double cpuMs) {
  if (!ps)
    return;
  op_profile_add_cpu(ps, name, cpuMs);
  ps->profile[name].isOuter = true;
  if (hipdnn_ep_trace_enabled())
    ps->trace.addComputeTotal(name, cpuStartUs, cpuMs * 1000.0);
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
    double gpuStartMs = -1.0; // epoch-relative start (for trace placement)
    if (ps->epoch) {
      float cumMs = 0.0f;
      hipError_t elErr = hipEventElapsedTime(&cumMs, ps->epoch,
                                             ps->markerPool[ev.markerIndex]);
      if (elErr != hipSuccess)
        cumMs = (float)tlPrevMs;
      gpuMs = (float)(cumMs - tlPrevMs);
      if (gpuMs < 0.0f)
        gpuMs = 0.0f;        // guard against event resolution jitter
      gpuStartMs = tlPrevMs; // op ran from prev marker to this one
      tlPrevMs = cumMs;
    }
    if (traceOn) {
      // Absolute timestamps on the shared axis. GPU start = inference's GPU
      // epoch (absolute) + this op's GPU-elapsed offset; CPU start = absolute
      // wrapper-entry time. Both already share the steady_clock trace axis, so
      // no per-track/per-inference re-basing is needed.
      double gTsUs = (gpuStartMs >= 0.0)
                         ? ps->gpuEpochAbsUs + gpuStartMs * 1000.0
                         : ev.cpuStartUs;
      ps->trace.addOp(ev.name, ev.shape, gTsUs, gpuMs * 1000.0, ev.cpuStartUs,
                      ev.cpuMs * 1000.0, ev.bytes);
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
    op.totalGpuMs += gpuMs;
    op.totalCpuMs += ev.cpuMs;
    op.totalCount++;
  }
  ps->pending.clear();

  // The trace file is written once, in op_profile_destroy (ChromeTrace::write
  // rewrites the whole file, so a per-inference write would be O(N^2); the JSON
  // is only consumed after the run). Here we just accumulate ops into
  // ps->trace.

  if (ps->profile.empty())
    return;

  struct OpRow {
    std::string name;
    bool hasGpu;
    bool isOuter;
    double totalGpuMs;
    double totalCpuMs;
    int64_t totalCount;
    std::vector<OpProfileState::ShapeEntry> shapes;
  };

  std::vector<OpRow> rows;
  int maxNameLen = 22;
  for (auto &[_, op] : ps->profile) {
    OpRow row;
    row.name = op.name;
    row.hasGpu = op.hasGpu;
    row.isOuter = op.isOuter;
    row.totalGpuMs = op.totalGpuMs;
    row.totalCpuMs = op.totalCpuMs;
    row.totalCount = op.totalCount;
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
    if (!r.isOuter)
      grandTotalCpuMs += r.totalCpuMs;
  }

  int lineWidth = maxNameLen + 2 + 5 + 1 + 9 + 1 + 9 + 1 + 6;
  fprintf(stderr, "\n[PERF] ");
  for (int i = 0; i < lineWidth; ++i)
    fputc('=', stderr);
  fputc('\n', stderr);
  fprintf(stderr, "[PERF]  %-*s %5s %9s %9s %6s\n", maxNameLen, "", "calls",
          "gpu (ms)", "cpu (ms)", "gpu %");

  for (auto &r : rows) {
    if (r.hasGpu) {
      double pct =
          grandTotalGpuMs > 0 ? r.totalGpuMs / grandTotalGpuMs * 100 : 0;
      fprintf(stderr, "[PERF]  %-*s %5lld %9.1f %9.1f %5.1f%%\n", maxNameLen,
              r.name.c_str(), (long long)r.totalCount, r.totalGpuMs,
              r.totalCpuMs, pct);
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
        fprintf(stderr, "[PERF]    %-*s %5lld %9.1f %9.1f %5.1f%%\n",
                maxNameLen - 2, sh.shape.c_str(), (long long)sh.count, sh.gpuMs,
                sh.cpuMs, pct);
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
