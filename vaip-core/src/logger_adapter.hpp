/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "glog/logging.h"
#include <chrono>
#include <memory>
#include <string>

namespace Ort {
struct Logger;
}

namespace vaip_core {

/**
 * @brief Set glog minimum log level from session option string
 *
 * This function provides a simplified interface for external callers.
 * Priority:
 * 1. session_option parameter (highest priority)
 * 2. ENV_PARAM(DEBUG_LOG_LEVEL) environment variable
 * 3. Default (ERROR level)
 *
 * @param session_option Optional session option string ("info", "warning",
 * "error", "fatal")
 */
void SetGlogMinLogLevel(const std::string& session_option);

class LoggerAdapter : public google::LogSink {
public:
  static std::shared_ptr<LoggerAdapter> get_current_logger();
  static std::shared_ptr<LoggerAdapter> create(const Ort::Logger& logger);
  LoggerAdapter(const Ort::Logger& logger);
  ~LoggerAdapter();

private:
  void send(google::LogSeverity severity, const char* full_filename,
            const char* base_filename, int line, const struct ::tm* tm_time,
            const char* message, size_t message_len) override final;

private:
  const Ort::Logger& logger_;
  const bool FLAGS_logtostderr_;
  const bool FLAGS_logtostdout_;
  const int FLAGS_minloglevel_;
};
} // namespace vaip_core
