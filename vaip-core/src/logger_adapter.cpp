/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define ORT_API_MANUAL_INIT
//
#include "./logger_adapter.hpp"
#include "core/graph/constants.h"
#include "morphizen/env_config.hpp"
#include "onnxruntime_cxx_api.h"
#include <cassert>
#include <glog/logging.h>

DEF_ENV_PARAM_2(DEBUG_LOG_LEVEL, "", std::string)

namespace vaip_core {

static int LogLevelStringToGlogSeverity(const std::string& level_str,
                                        int default_level) {
  if (level_str.empty()) {
    return default_level;
  }

  if (level_str == "info" || level_str == "INFO" || level_str == "verbose") {
    return google::GLOG_INFO;
  } else if (level_str == "warning" || level_str == "WARNING" ||
             level_str == "warn") {
    return google::GLOG_WARNING;
  } else if (level_str == "error" || level_str == "ERROR") {
    return google::GLOG_ERROR;
  } else if (level_str == "fatal" || level_str == "FATAL") {
    return google::GLOG_FATAL;
  } else {
    return default_level;
  }
}

static int OrtLoggingLevelToGlogSeverity(OrtLoggingLevel ort_level) {
  switch (ort_level) {
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
    return google::GLOG_INFO;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
    return google::GLOG_INFO;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
    return google::GLOG_WARNING;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:
    return google::GLOG_ERROR;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:
    return google::GLOG_FATAL;
  default:
    return google::GLOG_ERROR;
  }
}

static void SetGlogMinLogLevelWithOrt(const std::string& session_option,
                                      OrtLoggingLevel ort_level) {
  int log_level = google::GLOG_ERROR; // default
  std::string env_log_level = ENV_PARAM(DEBUG_LOG_LEVEL);

  // Priority 1: Check session_option parameter
  if (!session_option.empty()) {
    log_level =
        LogLevelStringToGlogSeverity(session_option, google::GLOG_ERROR);
  }
  // Priority 2: Check ENV_PARAM(DEBUG_LOG_LEVEL)
  else if (!env_log_level.empty()) {
    log_level = LogLevelStringToGlogSeverity(env_log_level, google::GLOG_ERROR);
  }
  // Priority 3: Use ORT logging level
  else if (ort_level != ORT_LOGGING_LEVEL_ERROR) {
    log_level = OrtLoggingLevelToGlogSeverity(ort_level);
  }

  FLAGS_minloglevel = log_level;
}

void SetGlogMinLogLevel(const std::string& session_option) {
  SetGlogMinLogLevelWithOrt(session_option, ORT_LOGGING_LEVEL_ERROR);
}

static std::weak_ptr<LoggerAdapter> g_the_current_logger;

std::shared_ptr<LoggerAdapter>
LoggerAdapter::create(const Ort::Logger& logger) {
  std::shared_ptr<LoggerAdapter> ret;
  if (g_the_current_logger.expired()) {
    ret = std::make_shared<LoggerAdapter>(logger);
    g_the_current_logger = ret;
  }
  ret = g_the_current_logger.lock();
  assert(ret != nullptr);
  return ret;
}

std::shared_ptr<LoggerAdapter> LoggerAdapter::get_current_logger() {
  return g_the_current_logger.lock();
}

LoggerAdapter::LoggerAdapter(const Ort::Logger& logger)
    : logger_(logger), FLAGS_logtostderr_(FLAGS_logtostderr),
      FLAGS_logtostdout_(FLAGS_logtostdout),
      FLAGS_minloglevel_(FLAGS_minloglevel) {
  // Initialize the logger
  FLAGS_logtostderr = 0;
  FLAGS_logtostdout = 0;
  FLAGS_minloglevel = 0;
  google::SetLogDestination(google::GLOG_INFO, "");
  google::SetLogDestination(google::GLOG_WARNING, "");
  google::SetLogDestination(google::GLOG_ERROR, "");
  google::SetLogDestination(google::GLOG_FATAL, "");
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(onnxruntime::kVitisAIExecutionProvider);
  }
  auto ort_logging_level = logger_.GetLoggingSeverityLevel();
  // Use unified log level interface with ORT level
  vaip_core::SetGlogMinLogLevelWithOrt("", ort_logging_level);
  google::AddLogSink(this);
}
LoggerAdapter::~LoggerAdapter() {
  // Cleanup the logger
  FLAGS_logtostderr = FLAGS_logtostderr_;
  FLAGS_logtostdout = FLAGS_logtostdout_;
  FLAGS_minloglevel = FLAGS_minloglevel_;
  google::RemoveLogSink(this);
}

void LoggerAdapter::send(google::LogSeverity glog_severity,
                         const char* /*full_filename */,
                         const char* base_filename, int line,
                         const struct ::tm* /*tm_time*/, const char* message,
                         size_t message_len) {
  auto ort_severity = logger_.GetLoggingSeverityLevel();

  switch (glog_severity) {
  case google::GLOG_INFO:
    ort_severity = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
    break;
  case google::GLOG_WARNING:
    ort_severity = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    break;
  case google::GLOG_ERROR:
    ort_severity = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
    break;
  case google::GLOG_FATAL:
    ort_severity = OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL;
    break;
  default:
    ort_severity = logger_.GetLoggingSeverityLevel();
    break;
  }
  /*
  const ORTCHAR_T* file_path =
      (const ORTCHAR_T*)"\0\0"; // for performance reasons, we don't translate

                                // file name
                                */
  const char* func_name = base_filename; // dirty hack, we assume glog
                                         // base_filename is the function name.
  // we relie on the fact that message is null terminated by glog.
  // tm is missing

  // Convert base_filename to ORT_CHAR_T*
  const ORTCHAR_T* ort_filename;
#ifdef _WIN32
  std::wstring wide_filename;
  if (base_filename) {
    size_t len = strlen(base_filename);
    wide_filename.resize(len);
    size_t converted = 0;
    mbstowcs_s(&converted, &wide_filename[0], len + 1, base_filename, len);
  }
  ort_filename = wide_filename.c_str();
#else
  ort_filename = base_filename;
#endif
  if (message[message_len] == '\n') {
    char* p = const_cast<char*>(message);
    p[message_len] =
        '\0'; // remove trailing '\n', it is glog::message::data_.message_text_,
              // it should be safe to modify it.
  }
  logger_.LogMessage(ort_severity, ort_filename, line, func_name, message);
}
} // namespace vaip_core
