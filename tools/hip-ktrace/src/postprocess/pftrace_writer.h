/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "common.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ktrace {
class PftraceWriter {
public:
  PftraceWriter(const fs::path &output, const fs::path &images,
                bool asynchronous);
  ~PftraceWriter();

  void Metadata(const std::string &json);
  void Event(const char *json, size_t size);
  void Close();
  uint64_t PacketCount() const { return packet_count_; }

private:
  struct Impl;
  using Batch = std::vector<std::string>;

  void Enqueue(Batch batch);
  void Worker();
  void Process(const std::string &json);

  std::ofstream output_;
  bool asynchronous_ = false;
  bool closed_ = false;
  Batch filling_;
  std::deque<Batch> pending_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable space_;
  bool stop_ = false;
  std::exception_ptr error_;
  std::thread worker_;
  uint64_t packet_count_ = 0;
  std::unique_ptr<Impl> impl_;
};
} // namespace ktrace
