/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <evntrace.h>

#include <filesystem>
#include <set>
#include <string>

namespace ktrace::collector {
class EtwCollector {
public:
  EtwCollector(std::filesystem::path output_dir, std::string prefix);
  ~EtwCollector();

  bool Start(std::string *error);
  bool StopAndExport(const std::set<std::uint32_t> &process_ids,
                     const std::set<std::uint32_t> &thread_ids,
                     std::string *error);
  bool Enabled() const { return session_ != 0; }

private:
  bool Parse(const std::set<std::uint32_t> &process_ids,
             const std::set<std::uint32_t> &thread_ids, std::string *error);

  std::filesystem::path output_dir_;
  std::string prefix_;
  std::wstring session_name_;
  std::filesystem::path etl_path_;
  TRACEHANDLE session_ = 0;
};
} // namespace ktrace::collector
