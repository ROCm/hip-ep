/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Chrome trace (chrome://tracing / Perfetto) for the EP op profiler. One
// instance per EP session: it accumulates per-op GPU/CPU spans and pipeline
// (H2D/Compute/D2H) spans, then writes a per-session JSON file. All timestamps
// are absolute microseconds on the shared steady_clock axis (see op_profile.cpp
// for how events are timed), so concurrent sessions merge without re-basing.
class ChromeTrace {
public:
  // Builds a globally-unique session token (pid + this) used for the file name
  // and process label, so concurrent sessions never clobber each other -- no
  // shared counter, which matters because the runtime is statically linked into
  // each per-model NATIVE model.dll.
  ChromeTrace();

  // Per-op spans: GPU track (epoch-anchored) + CPU track (wrapper wall time).
  void addOp(const std::string &name, const std::string &shape, double gpuTsUs,
             double gpuDurUs, double cpuTsUs, double cpuDurUs, int64_t bytes);

  // One inference's pipeline phase spans, laid end-to-end from `epochAbsUs`.
  void addIoSpans(double epochAbsUs, double h2dMs, int64_t h2dBytes,
                  double computeMs, double d2hMs, int64_t d2hBytes);

  // Outer whole-Compute CPU span (steady_clock) placed on the CPU (wrapper)
  // track so the per-op CPU spans nest inside it. The uncovered width is the
  // "bubble" -- host time not attributed to any op wrapper (input marshaling,
  // per-kernel launch/dispatch, inter-op gaps, the post-compute fence).
  void addComputeTotal(const std::string &name, double cpuTsUs,
                       double cpuDurUs);

  // (Re)write the whole per-session file: `basePath` -> `basePath.<tag>.ext`.
  // `rooflineGbps` drives the per-op GB/s and %peak args (<=0 => omitted as 0).
  void write(const std::string &basePath, double rooflineGbps) const;

private:
  struct Op {
    std::string name, shape;
    double tsUs, durUs;       // GPU track
    double cpuTsUs, cpuDurUs; // CPU track
    int64_t bytes;
  };
  struct IoSpan {
    std::string name;
    int tid;
    double tsUs, durUs;
    int64_t bytes;
  };
  struct CpuTotal {
    std::string name;
    double tsUs, durUs;
  };

  std::string tag_;
  std::vector<Op> ops_;
  std::vector<IoSpan> io_;
  std::vector<CpuTotal> cpuTotals_;
};
