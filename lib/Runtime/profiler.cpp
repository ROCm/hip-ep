/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "profiler.h"
#include "real/runtime_types.h"

// Single definition of the static member (not inline — avoids
// per-module duplication when llvm-linked across bitcode modules).
PerfState *PerfState::instance_ = nullptr;

PerfTimer::~PerfTimer() {
  if (!active_)
    return;
  // Sync stream to ensure GPU work is complete before measuring
  if (stream_)
    (void)hipStreamSynchronize(stream_);
  auto end = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(end - start_).count();
  PerfState::get()->record(op_, shape_, ms);
}
