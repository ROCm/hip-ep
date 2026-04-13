/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <chrono>
#include <cstdlib>

namespace hip::profiling {

inline bool profilingEnabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_PROFILE");
    return v && v[0] >= '1';
  }();
  return enabled;
}

inline int &currentDepth() {
  static thread_local int depth = 0;
  return depth;
}

class ScopedTimer {
public:
  explicit ScopedTimer(const char *label, bool enabled = profilingEnabled())
      : label_(label), enabled_(enabled) {
    if (!enabled_)
      return;
    depth_ = currentDepth()++;
    start_ = std::chrono::steady_clock::now();
  }

  ~ScopedTimer() {
    if (!enabled_)
      return;
    --currentDepth();
    auto elapsed = std::chrono::steady_clock::now() - start_;
    double ms =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
        1000.0;
    llvm::errs() << "[PROFILE]";
    for (int i = 0; i < depth_; ++i)
      llvm::errs() << "  ";
    llvm::errs() << " " << label_ << ": " << llvm::format("%.1f", ms)
                 << " ms\n";
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
  const char *label_;
  bool enabled_;
  int depth_ = 0;
  std::chrono::steady_clock::time_point start_;
};

} // namespace hip::profiling

#define HIP_PROFILE_SCOPE(label)                                               \
  ::hip::profiling::ScopedTimer _hip_timer_##__LINE__(label)
