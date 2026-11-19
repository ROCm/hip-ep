/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "ktrace_ipc.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace ktrace::collector {
class SpoolWriter {
public:
  SpoolWriter(std::filesystem::path output_dir, std::string prefix,
              std::uint64_t resume_timestamp_ns);
  void Write(const ipc::Record &record);
  void Finish(std::uint64_t pushed, std::uint64_t dropped,
              std::uint64_t high_watermark);
  const std::set<std::uint32_t> &ProcessIds() const { return process_ids_; }
  const std::set<std::uint32_t> &ThreadIds() const { return thread_ids_; }

private:
  struct Files {
    std::uint32_t pid = 0;
    std::uint64_t op_id = 0;
    std::uint64_t ops = 0;
    std::uint64_t apis = 0;
    std::uint64_t roctx = 0;
    ipc::DevicePayload device{};
    bool have_device = false;
    std::ofstream ops_file;
    std::ofstream api_file;
    std::ofstream roctx_file;
    std::ofstream vram_file;
    std::ofstream kernel_file;
  };

  Files &Process(std::uint32_t pid);
  std::filesystem::path Base(std::uint32_t pid) const;
  void EnsureOps(Files &files);
  void WriteSummary(Files &files, std::uint64_t pushed, std::uint64_t dropped,
                    std::uint64_t high_watermark);

  std::filesystem::path output_dir_;
  std::string prefix_;
  std::uint64_t resume_timestamp_ns_ = 0;
  std::map<std::uint32_t, Files> processes_;
  std::set<std::uint32_t> process_ids_;
  std::set<std::uint32_t> thread_ids_;
};
} // namespace ktrace::collector
