/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "ktrace_ipc.h"

#include <cstdio>
#include <filesystem>
#include <vector>

namespace ktrace::collector {
class SpoolWriter;

class BinaryJournal {
public:
  explicit BinaryJournal(std::filesystem::path path);
  ~BinaryJournal();

  void Write(const ipc::Record &record);
  void Replay(SpoolWriter *writer);
  void Remove();

private:
  void Close();

  std::filesystem::path path_;
  std::vector<char> buffer_;
  FILE *file_ = nullptr;
};
} // namespace ktrace::collector
