/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "glog/logging.h"
#include <chrono>
#include <memory>
namespace Ort {
struct Logger;
}
namespace morphizen {
class LoggerAdapter : public google::LogSink {
public:
  LoggerAdapter(std::unique_ptr<Ort::Logger> logger);
  ~LoggerAdapter();

private:
  void send(google::LogSeverity severity, const char *full_filename,
            const char *base_filename, int line, const struct ::tm *tm_time,
            const char *message, size_t message_len) override final;

private:
  std::unique_ptr<Ort::Logger> logger_;
  const bool FLAGS_logtostderr_;
  const bool FLAGS_logtostdout_;
  const int FLAGS_minloglevel_;
};
} // namespace morphizen
