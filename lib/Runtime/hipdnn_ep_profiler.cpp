/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipdnn_ep_profiler.cpp - GPU Profiling Implementation --------------===//
//
// hipEvent-based GPU profiling for runtime wrapper functions.
//
// When enabled, records hipEvents at PROFILE_BEGIN / PROFILE_END boundaries.
// On dump, synchronizes the stream, computes elapsed times, and writes:
//   1. hipdnn_ep_profile.json  (Chrome Tracing format)
//   2. hipdnn_ep_profile.csv   (name,category,start_us,duration_us,duration_ms)
//   3. Summary table to stderr (sorted by total time descending)
//
//===----------------------------------------------------------------------===//

#ifdef HIPDNN_EP_ENABLE_PROFILING

#include "hipdnn_ep_profiler.h"
#include "runtime_state_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

//===----------------------------------------------------------------------===//
// Internal data structures
//===----------------------------------------------------------------------===//

struct ProfileEvent {
  const char *name;
  const char *category;
  void *event_start; // hipEvent_t
  void *event_end;   // hipEvent_t
  int64_t host_ts_us; // host timestamp at begin (microseconds since profiler start)
  bool has_end;
};

struct ProfilerState {
  bool enabled;
  ProfileEvent *events;
  size_t event_count;
  size_t event_capacity;
  std::chrono::steady_clock::time_point start_time;
};

static constexpr size_t kDefaultCapacity = 16384;

//===----------------------------------------------------------------------===//
// Profiler API implementation
//===----------------------------------------------------------------------===//

void hipdnn_ep_profiler_init(RuntimeState *state) {
  if (!state)
    return;

  const char *env = std::getenv("HIPDNN_EP_PROFILE");
  if (!env || env[0] == '\0' || env[0] == '0') {
    state->profiler = nullptr;
    return;
  }

  ProfilerState *ps =
      static_cast<ProfilerState *>(std::malloc(sizeof(ProfilerState)));
  if (!ps) {
    fprintf(stderr, "[PROFILER] Failed to allocate ProfilerState\n");
    state->profiler = nullptr;
    return;
  }

  ps->enabled = true;
  ps->event_count = 0;
  ps->event_capacity = kDefaultCapacity;
  ps->events = static_cast<ProfileEvent *>(
      std::calloc(kDefaultCapacity, sizeof(ProfileEvent)));
  if (!ps->events) {
    fprintf(stderr, "[PROFILER] Failed to allocate events array\n");
    std::free(ps);
    state->profiler = nullptr;
    return;
  }
  ps->start_time = std::chrono::steady_clock::now();

  state->profiler = ps;
  fprintf(stderr, "[PROFILER] Initialized (capacity=%zu)\n",
          ps->event_capacity);
}

void hipdnn_ep_profiler_begin(RuntimeState *state, const char *name,
                              const char *category) {
  if (!state || !state->profiler)
    return;

  ProfilerState *ps = static_cast<ProfilerState *>(state->profiler);
  if (!ps->enabled)
    return;

  if (ps->event_count >= ps->event_capacity) {
    fprintf(stderr,
            "[PROFILER] WARNING: event capacity (%zu) exceeded, dropping '%s'\n",
            ps->event_capacity, name);
    return;
  }

  hipEvent_t ev_start = nullptr;
  if (hipEventCreate(&ev_start) != hipSuccess) {
    fprintf(stderr, "[PROFILER] hipEventCreate failed for '%s'\n", name);
    return;
  }

  if (hipEventRecord(ev_start, state->stream) != hipSuccess) {
    fprintf(stderr, "[PROFILER] hipEventRecord failed for '%s'\n", name);
    hipEventDestroy(ev_start);
    return;
  }

  auto now = std::chrono::steady_clock::now();
  int64_t host_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        now - ps->start_time)
                        .count();

  ProfileEvent &e = ps->events[ps->event_count++];
  e.name = name;
  e.category = category;
  e.event_start = ev_start;
  e.event_end = nullptr;
  e.host_ts_us = host_us;
  e.has_end = false;
}

void hipdnn_ep_profiler_end(RuntimeState *state, const char *name,
                            const char *category) {
  if (!state || !state->profiler)
    return;

  ProfilerState *ps = static_cast<ProfilerState *>(state->profiler);
  if (!ps->enabled)
    return;

  // Search backwards for matching begin event
  ProfileEvent *match = nullptr;
  for (size_t i = ps->event_count; i > 0; --i) {
    ProfileEvent &e = ps->events[i - 1];
    if (!e.has_end && std::strcmp(e.name, name) == 0) {
      match = &e;
      break;
    }
  }

  if (!match) {
    fprintf(stderr,
            "[PROFILER] WARNING: no matching begin for end '%s', ignoring\n",
            name);
    return;
  }

  hipEvent_t ev_end = nullptr;
  if (hipEventCreate(&ev_end) != hipSuccess) {
    fprintf(stderr, "[PROFILER] hipEventCreate (end) failed for '%s'\n", name);
    return;
  }

  if (hipEventRecord(ev_end, state->stream) != hipSuccess) {
    fprintf(stderr, "[PROFILER] hipEventRecord (end) failed for '%s'\n", name);
    hipEventDestroy(ev_end);
    return;
  }

  match->event_end = ev_end;
  match->has_end = true;
}

//===----------------------------------------------------------------------===//
// Dump: synchronize, compute elapsed times, write outputs
//===----------------------------------------------------------------------===//

/// Summary row for aggregated per-name statistics
struct SummaryRow {
  const char *name;
  const char *category;
  int calls;
  double total_ms;
  double avg_ms;
};

void hipdnn_ep_profiler_dump(RuntimeState *state) {
  if (!state || !state->profiler)
    return;

  ProfilerState *ps = static_cast<ProfilerState *>(state->profiler);
  if (!ps->enabled || ps->event_count == 0)
    return;

  // Synchronize stream to ensure all events have completed
  if (state->stream) {
    hipStreamSynchronize(state->stream);
  }

  // Compute elapsed times for all events
  struct ResolvedEvent {
    const char *name;
    const char *category;
    int64_t host_ts_us;
    float duration_ms;
  };

  std::vector<ResolvedEvent> resolved;
  resolved.reserve(ps->event_count);

  for (size_t i = 0; i < ps->event_count; ++i) {
    ProfileEvent &e = ps->events[i];
    if (!e.has_end || !e.event_start || !e.event_end)
      continue;

    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, static_cast<hipEvent_t>(e.event_start),
                            static_cast<hipEvent_t>(e.event_end)) !=
        hipSuccess) {
      continue;
    }

    resolved.push_back(
        {e.name, e.category, e.host_ts_us, elapsed_ms});
  }

  if (resolved.empty()) {
    fprintf(stderr, "[PROFILER] No completed events to dump\n");
    return;
  }

  //===--------------------------------------------------------------------===//
  // 1. Write Chrome Tracing JSON
  //===--------------------------------------------------------------------===//
  {
    FILE *fp = fopen("hipdnn_ep_profile.json", "w");
    if (fp) {
      fprintf(fp, "{\"traceEvents\":[\n");
      for (size_t i = 0; i < resolved.size(); ++i) {
        auto &r = resolved[i];
        double dur_us = static_cast<double>(r.duration_ms) * 1000.0;
        fprintf(fp,
                "  {\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\","
                "\"ts\":%lld,\"dur\":%.1f,\"pid\":1,\"tid\":1}%s\n",
                r.name, r.category, (long long)r.host_ts_us, dur_us,
                (i + 1 < resolved.size()) ? "," : "");
      }
      fprintf(fp, "]}\n");
      fclose(fp);
      fprintf(stderr, "[PROFILER] Wrote hipdnn_ep_profile.json (%zu events)\n",
              resolved.size());
    } else {
      fprintf(stderr, "[PROFILER] WARNING: failed to open "
                      "hipdnn_ep_profile.json for writing\n");
    }
  }

  //===--------------------------------------------------------------------===//
  // 2. Write CSV
  //===--------------------------------------------------------------------===//
  {
    FILE *fp = fopen("hipdnn_ep_profile.csv", "w");
    if (fp) {
      fprintf(fp, "name,category,start_us,duration_us,duration_ms\n");
      for (auto &r : resolved) {
        double dur_us = static_cast<double>(r.duration_ms) * 1000.0;
        fprintf(fp, "%s,%s,%lld,%.1f,%.4f\n", r.name, r.category,
                (long long)r.host_ts_us, dur_us, r.duration_ms);
      }
      fclose(fp);
      fprintf(stderr, "[PROFILER] Wrote hipdnn_ep_profile.csv (%zu events)\n",
              resolved.size());
    } else {
      fprintf(stderr, "[PROFILER] WARNING: failed to open "
                      "hipdnn_ep_profile.csv for writing\n");
    }
  }

  //===--------------------------------------------------------------------===//
  // 3. Summary table to stderr
  //===--------------------------------------------------------------------===//
  {
    // Aggregate by name
    std::vector<SummaryRow> summary;
    for (auto &r : resolved) {
      bool found = false;
      for (auto &s : summary) {
        if (std::strcmp(s.name, r.name) == 0) {
          s.calls++;
          s.total_ms += r.duration_ms;
          found = true;
          break;
        }
      }
      if (!found) {
        SummaryRow row;
        row.name = r.name;
        row.category = r.category;
        row.calls = 1;
        row.total_ms = r.duration_ms;
        row.avg_ms = 0.0;
        summary.push_back(row);
      }
    }

    // Compute total and averages
    double grand_total_ms = 0.0;
    for (auto &s : summary) {
      s.avg_ms = s.total_ms / s.calls;
      grand_total_ms += s.total_ms;
    }

    // Sort by total time descending
    std::sort(summary.begin(), summary.end(),
              [](const SummaryRow &a, const SummaryRow &b) {
                return a.total_ms > b.total_ms;
              });

    fprintf(stderr,
            "\n[PROFILER] ==================== GPU Profile Summary "
            "====================\n");
    fprintf(stderr, "%-40s %8s %12s %12s %8s\n", "Function", "Calls",
            "Total (ms)", "Avg (ms)", "%%");
    fprintf(stderr,
            "----------------------------------------------------------------------"
            "----------\n");

    for (auto &s : summary) {
      double pct = (grand_total_ms > 0.0) ? (s.total_ms / grand_total_ms * 100.0)
                                           : 0.0;
      fprintf(stderr, "%-40s %8d %12.3f %12.3f %7.1f%%\n", s.name, s.calls,
              s.total_ms, s.avg_ms, pct);
    }

    fprintf(stderr,
            "----------------------------------------------------------------------"
            "----------\n");
    fprintf(stderr, "%-40s %8s %12.3f\n", "TOTAL", "",
            grand_total_ms);
    fprintf(stderr,
            "======================================================================"
            "==========\n\n");
  }
}

//===----------------------------------------------------------------------===//
// Cleanup: destroy all hipEvents, free memory
//===----------------------------------------------------------------------===//

void hipdnn_ep_profiler_cleanup(RuntimeState *state) {
  if (!state || !state->profiler)
    return;

  ProfilerState *ps = static_cast<ProfilerState *>(state->profiler);

  for (size_t i = 0; i < ps->event_count; ++i) {
    ProfileEvent &e = ps->events[i];
    if (e.event_start) {
      hipEventDestroy(static_cast<hipEvent_t>(e.event_start));
    }
    if (e.event_end) {
      hipEventDestroy(static_cast<hipEvent_t>(e.event_end));
    }
  }

  std::free(ps->events);
  std::free(ps);
  state->profiler = nullptr;

  fprintf(stderr, "[PROFILER] Cleanup complete\n");
}

#endif /* HIPDNN_EP_ENABLE_PROFILING */
