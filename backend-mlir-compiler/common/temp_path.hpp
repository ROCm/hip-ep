/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <process.h>
#define MORPHIZEN_GETPID() _getpid()
#else
#include <unistd.h>
#define MORPHIZEN_GETPID() ::getpid()
#endif

namespace mlir_compiler_utils {

// Generate a collision-resistant temporary file path.
// Combines OS temp directory, PID, millisecond timestamp, and an atomic
// counter to avoid races across processes, threads, and rapid successive calls.
// suffix should include the file extension, e.g. ".dll", or "" for no
// extension.
inline std::string generateTempPath(const std::string &suffix) {
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();

  static std::atomic<int> counter{0};
  int pid = static_cast<int>(MORPHIZEN_GETPID());

  std::string filename = "morphizen_mlir_" + std::to_string(pid) + "_" +
                         std::to_string(timestamp) + "_" +
                         std::to_string(counter++) + suffix;

  return (std::filesystem::temp_directory_path() / filename).string();
}

} // namespace mlir_compiler_utils
