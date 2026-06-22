/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "op_profile.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

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
    std::map<std::string, ShapeEntry> shapes;
    double totalGpuMs = 0;
    double totalCpuMs = 0;
    int64_t totalCount = 0;
  };
  struct PendingEvent {
    std::string name;
    std::string shape;
    int eventIndex;
    double cpuMs;
  };

  // Pre-allocated event pool: each pair is (start, stop).
  // Grows on demand but never shrinks — avoids hipEventCreate/Destroy per op.
  struct EventPair {
    hipEvent_t start;
    hipEvent_t stop;
  };
  std::vector<EventPair> eventPool;
  int nextEventIndex = 0;

  std::map<std::string, OpEntry> profile;
  std::vector<PendingEvent> pending;
  bool active = false;
};

OpProfileState *op_profile_create() { return new OpProfileState; }

void op_profile_destroy(OpProfileState *ps) {
  if (!ps)
    return;
  for (auto &ep : ps->eventPool) {
    hipEventDestroy(ep.start);
    hipEventDestroy(ep.stop);
  }
  delete ps;
}

void op_profile_reset(OpProfileState *ps) {
  if (!ps)
    return;
  ps->profile.clear();
  ps->pending.clear();
  ps->nextEventIndex = 0;
  ps->active = true;
}

bool op_profile_is_active(OpProfileState *ps) { return ps && ps->active; }

int op_profile_acquire_event_pair(OpProfileState *ps) {
  int idx = ps->nextEventIndex++;
  if (idx >= (int)ps->eventPool.size()) {
    OpProfileState::EventPair ep;
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
    hipEventCreateWithFlags(&ep.start, hipEventDisableSystemFence);
    hipEventCreateWithFlags(&ep.stop, hipEventDisableSystemFence);
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

void op_profile_add_pending(OpProfileState *ps, const std::string &name,
                            const std::string &shape, int eventIndex,
                            double cpuMs) {
  if (!ps)
    return;
  ps->pending.push_back({name, shape, eventIndex, cpuMs});
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

void op_profile_resolve_and_print(OpProfileState *ps) {
  if (!ps)
    return;

  for (auto &ev : ps->pending) {
    float gpuMs = 0.0f;
    hipEventElapsedTime(&gpuMs, ps->eventPool[ev.eventIndex].start,
                        ps->eventPool[ev.eventIndex].stop);
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

  if (ps->profile.empty())
    return;

  struct OpRow {
    std::string name;
    bool hasGpu;
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
    grandTotalCpuMs += r.totalCpuMs;
  }

  int lineWidth = maxNameLen + 2 + 5 + 1 + 9 + 1 + 9 + 1 + 6;
  // Emit the separator as a single formatted line (no CRT stdio FILE*): the
  // bitcode-JIT'd runtime resolves fputc to a different CRT than the stdout/
  // stderr FILE table, so a direct fputc(stderr) is a cross-CRT FILE* handoff
  // (corruption). hipdnn_ep_log_emit goes straight to the OS stderr handle.
  hipdnn_ep_log_emit("\n[PERF] %s\n", std::string(lineWidth, '=').c_str());
  hipdnn_ep_log_emit("[PERF]  %-*s %5s %9s %9s %6s\n", maxNameLen, "", "calls",
                     "gpu (ms)", "cpu (ms)", "gpu %");

  for (auto &r : rows) {
    if (r.hasGpu) {
      double pct =
          grandTotalGpuMs > 0 ? r.totalGpuMs / grandTotalGpuMs * 100 : 0;
      hipdnn_ep_log_emit("[PERF]  %-*s %5lld %9.1f %9.1f %5.1f%%\n", maxNameLen,
                         r.name.c_str(), (long long)r.totalCount, r.totalGpuMs,
                         r.totalCpuMs, pct);
    } else {
      hipdnn_ep_log_emit("[PERF]  %-*s %5lld %9s %9.1f %6s\n", maxNameLen,
                         r.name.c_str(), (long long)r.totalCount, "n/a",
                         r.totalCpuMs, "n/a");
    }
    bool hasShapes = r.shapes.size() > 1 ||
                     (r.shapes.size() == 1 && !r.shapes[0].shape.empty());
    if (hasShapes) {
      for (auto &sh : r.shapes) {
        if (sh.shape.empty())
          continue;
        double pct = grandTotalGpuMs > 0 ? sh.gpuMs / grandTotalGpuMs * 100 : 0;
        hipdnn_ep_log_emit("[PERF]    %-*s %5lld %9.1f %9.1f %5.1f%%\n",
                           maxNameLen - 2, sh.shape.c_str(),
                           (long long)sh.count, sh.gpuMs, sh.cpuMs, pct);
      }
    }
  }
  hipdnn_ep_log_emit("[PERF]  %-*s %5s %9.1f %9.1f\n", maxNameLen, "TOTAL", "",
                     grandTotalGpuMs, grandTotalCpuMs);
  hipdnn_ep_log_emit("[PERF] %s\n\n", std::string(lineWidth, '=').c_str());
  ps->profile.clear();
}
