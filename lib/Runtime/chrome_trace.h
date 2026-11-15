/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Typed diagnostic JSON. There is deliberately no raw-JSON or pointer
// constructor: metadata describes actual views/control values, never tensor
// contents or addresses. Missing values remain explicit nulls.
class CallCaptureJson {
public:
  CallCaptureJson() : text_("null") {}
  CallCaptureJson(std::nullptr_t) : text_("null") {}
  CallCaptureJson(bool value) : text_(value ? "true" : "false") {}

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value &&
                                        !std::is_same<T, bool>::value,
                                    int>::type = 0>
  CallCaptureJson(T value) {
    char buffer[32];
    if constexpr (std::is_signed<T>::value)
      std::snprintf(buffer, sizeof(buffer), "%lld",
                    static_cast<long long>(value));
    else
      std::snprintf(buffer, sizeof(buffer), "%llu",
                    static_cast<unsigned long long>(value));
    text_ = buffer;
  }

  CallCaptureJson(double value) {
    if (!std::isfinite(value)) {
      text_ = "null";
      return;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    text_ = buffer;
  }

  CallCaptureJson(const std::string &value) : text_(quote(value)) {}
  CallCaptureJson(const char *value) : text_(value ? quote(value) : "null") {}
  CallCaptureJson(const void *) = delete;

  static CallCaptureJson array(const std::vector<CallCaptureJson> &values) {
    CallCaptureJson result;
    result.text_ = "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        result.text_ += ',';
      result.text_ += values[i].text_;
    }
    result.text_ += ']';
    return result;
  }

  static CallCaptureJson dimensions(const std::vector<int64_t> &values) {
    std::vector<CallCaptureJson> dimensions;
    dimensions.reserve(values.size());
    for (int64_t value : values)
      dimensions.emplace_back(value);
    return array(dimensions);
  }

  static CallCaptureJson object(
      std::initializer_list<std::pair<std::string, CallCaptureJson>> values) {
    CallCaptureJson result;
    result.text_ = "{";
    bool first = true;
    for (const auto &value : values) {
      if (!first)
        result.text_ += ',';
      first = false;
      result.text_ += quote(value.first) + ':' + value.second.text_;
    }
    result.text_ += '}';
    return result;
  }

  // Add a producer-owned identity to an object without accepting raw JSON.
  CallCaptureJson withIdentity(const char *field,
                               const std::string &identity) const {
    if (text_.empty() || text_[0] != '{')
      return {};
    CallCaptureJson result;
    result.text_ = "{" + quote(field) + ':' + quote(identity);
    if (text_.size() > 2)
      result.text_ += ',';
    result.text_ += text_.substr(1);
    return result;
  }

  const std::string &text() const { return text_; }

private:
  static std::string quote(const std::string &value) {
    std::string result = "\"";
    for (unsigned char character : value) {
      switch (character) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (character < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
          result += buffer;
        } else {
          result += static_cast<char>(character);
        }
      }
    }
    result += '"';
    return result;
  }

  std::string text_;
};

// Supplied at the actual call boundary. The trace transport cannot infer any
// missing graph/window/operand association from a shape label or event order.
struct OpCallCapture {
  bool exampleOnly = false;
  std::string callId;
  std::string caseId;
  std::string graphId;
  std::string windowId;
  std::string stage;
  CallCaptureJson descriptor;
  CallCaptureJson graph;
  CallCaptureJson window;
};

// Chrome trace (chrome://tracing / Perfetto) for the EP op profiler. One
// instance per EP session: it accumulates per-op GPU/CPU spans and pipeline
// (H2D/Compute/D2H) spans, then writes a per-session JSON file. All timestamps
// are absolute microseconds on the shared steady_clock axis (see op_profile.cpp
// for how events are timed), so concurrent sessions merge without re-basing.
class ChromeTrace {
public:
  // Builds a token for concurrent live instances. Captured sessions also use
  // the loader's process-wide identity: addresses can recur after destruction,
  // and counters local to each NATIVE model.dll cannot distinguish reloads.
  ChromeTrace();

  void setSessionCapture(const std::string &sessionId,
                         const CallCaptureJson &capture) {
    tag_ = "p" + std::to_string(currentProcessId_) + "_" + sessionId;
    sessionCapture_ = capture;
  }

  // Per-op spans: GPU track (epoch-anchored) + CPU track (wrapper wall time).
  void addOp(const std::string &name, const std::string &shape, double gpuTsUs,
             double gpuDurUs, double cpuTsUs, double cpuDurUs, int64_t bytes,
             const OpCallCapture *capture = nullptr,
             const CallCaptureJson *executionContext = nullptr);

  // One inference's pipeline phase spans, laid end-to-end from `epochAbsUs`.
  void addIoSpans(double epochAbsUs, double h2dMs, int64_t h2dBytes,
                  double computeMs, double d2hMs, int64_t d2hBytes);

  // Outer whole-Compute CPU span (steady_clock) placed on the CPU (wrapper)
  // track so the per-op CPU spans nest inside it. The uncovered width is the
  // "bubble" -- host time not attributed to any op wrapper (input marshaling,
  // per-kernel launch/dispatch, inter-op gaps, the post-compute fence).
  void addComputeTotal(const std::string &name, double cpuTsUs,
                       double cpuDurUs);

  // (Re)write the whole per-session file: `basePath` -> `basePath.<tag>.ext`.
  // `rooflineGbps` drives the per-op GB/s and %peak args (<=0 => omitted as 0).
  void write(const std::string &basePath, double rooflineGbps) const;

private:
  struct Op {
    std::string name, shape;
    double tsUs, durUs;       // GPU track
    double cpuTsUs, cpuDurUs; // CPU track
    int64_t bytes;
    std::optional<OpCallCapture> capture;
    std::optional<CallCaptureJson> executionContext;
  };
  struct IoSpan {
    std::string name;
    int tid;
    double tsUs, durUs;
    int64_t bytes;
  };
  struct CpuTotal {
    std::string name;
    double tsUs, durUs;
  };

  std::string tag_;
  long currentProcessId_ = 0;
  std::optional<CallCaptureJson> sessionCapture_;
  std::vector<Op> ops_;
  std::vector<IoSpan> io_;
  std::vector<CpuTotal> cpuTotals_;
};
