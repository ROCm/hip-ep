/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define ORT_API_MANUAL_INIT
//
#include "./logger_adapter.hpp"
#include "onnxruntime_cxx_api.h"
#include <cassert>

// Define the constant directly instead of including internal ONNX Runtime
// headers. This constant was originally in core/graph/constants.h (internal
// header not available in installed ONNX Runtime packages). The string
// "MorphiZenExecutionProvider" is a stable EP identifier defined by ONNX
// Runtime plugin EP specification and cannot change without breaking the plugin
// EP mechanism.
namespace onnxruntime {
constexpr const char* kMorphiZenExecutionProvider =
    "MorphiZenExecutionProvider";
}

namespace morphizen {

LoggerAdapter::LoggerAdapter(std::unique_ptr<Ort::Logger> logger)
    : logger_(std::move(logger)), FLAGS_logtostderr_(FLAGS_logtostderr),
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
    google::InitGoogleLogging(onnxruntime::kMorphiZenExecutionProvider);
  }
  auto ort_logging_level = logger_->GetLoggingSeverityLevel();
  switch (ort_logging_level) {
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
    FLAGS_minloglevel = google::GLOG_INFO;
    break;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
    FLAGS_minloglevel = google::GLOG_INFO;
    break;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
    FLAGS_minloglevel = google::GLOG_WARNING;
    break;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:
    FLAGS_minloglevel = google::GLOG_ERROR;
    break;
  case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:
    FLAGS_minloglevel = google::GLOG_FATAL;
    break;
  default:
    FLAGS_minloglevel = FLAGS_minloglevel_;
    break;
  }
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
  auto ort_severity = logger_->GetLoggingSeverityLevel();

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
    ort_severity = logger_->GetLoggingSeverityLevel();
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
  logger_->LogMessage(ort_severity, ort_filename, line, func_name, message);
}
} // namespace morphizen
