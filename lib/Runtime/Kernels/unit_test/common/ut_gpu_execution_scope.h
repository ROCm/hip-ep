/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace hipdnn_ep_test {
// Optional launcher handshake, outside timing. Construct after CPU reference
// preparation and before device work. Declare before device-owning objects so
// their cleanup completes before the launcher releases its execution lease.
class GpuExecutionScope {
 public:
  explicit GpuExecutionScope(std::function<void()> synchronize) : synchronize_(synchronize) {
    const char* prefix = std::getenv("HIPDNN_UT_GPU_ADMISSION");
    if (!prefix || !prefix[0]) return;
    static unsigned next = 0;
    path_ = std::string(prefix) + "." + std::to_string(next++);
    signal(".request");
    wait(".grant");
  }
  GpuExecutionScope(const GpuExecutionScope&) = delete;
  GpuExecutionScope& operator=(const GpuExecutionScope&) = delete;
  ~GpuExecutionScope() {
    if (path_.empty()) return;
    // A synchronization error still requires the launcher to reconcile the
    // ended worker, rather than retaining or releasing a live GPU worker.
    try {
      synchronize_();
      signal(".done");
      wait(".released");
    } catch (...) {
      std::terminate();
    }
  }

 private:
  std::string path_;
  std::function<void()> synchronize_;
  void signal(const char* suffix) {
    std::ofstream out(path_ + suffix);
    if (!out) throw std::runtime_error("UT GPU admission: cannot signal launcher");
    out << "ready\n";
    out.close();
    if (!out) throw std::runtime_error("UT GPU admission: signal write failed");
  }
  void wait(const char* suffix) {
    while (!std::ifstream(path_ + suffix).good())
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
};
}  // namespace hipdnn_ep_test
