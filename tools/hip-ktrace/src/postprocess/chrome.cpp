/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"
#include "pftrace_writer.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ktrace {
namespace {
constexpr int kGpuPid = 0;
constexpr int kHostPid = 1;
constexpr int kRoctxPid = 2;
constexpr int kPhasePid = 3;
constexpr int kCpuPid = 4;
constexpr int kCpuSamplePid = 5;

const std::unordered_set<std::string> kBlockingApis = {"hipStreamSynchronize",
                                                       "hipEventSynchronize",
                                                       "hipDeviceSynchronize",
                                                       "hipStreamWaitEvent",
                                                       "hipMemcpy",
                                                       "hipMalloc",
                                                       "hipFree",
                                                       "hipModuleLoad"};

struct Options {
  fs::path source;
  fs::path output;
  fs::path pftrace_output;
  std::optional<double> start_ms;
  double span_ms = 500.0;
  bool all = false;
  std::string phase;
  bool no_flows = false;
  bool force_flows = false;
  bool json_requested = false;
  bool pftrace_requested = false;
  bool format_requested = false;
  bool quiet = false;
};

struct OpRow {
  std::string id;
  std::string op;
  std::string kind;
  std::string kernel;
  uint64_t bytes = 0;
  std::string device;
  int64_t queue = 0;
  int64_t stream = 0;
  int64_t thread = 0;
  uint64_t corr = 0;
  double begin_us = 0;
  double end_us = 0;
};

struct OpTiming {
  int64_t queue = 0;
  double begin_us = 0;
  double end_us = 0;
};

struct LanePlan {
  std::vector<uint32_t> by_op;
  std::unordered_map<int64_t, uint32_t> counts;
};

struct ApiRow {
  uint64_t corr = 0;
  std::string api;
  int64_t thread = 0;
  double enter_us = 0;
  double exit_us = 0;
  std::string args;
};

struct RoctxRow {
  std::string operation;
  int64_t process = 0;
  int64_t thread = 0;
  uint64_t range_id = 0;
  double timestamp_us = 0;
  std::string message;
};

struct KernelAttrs {
  std::string occupancy;
  std::string vgpr;
  std::string scratch;
  std::string static_lds;
};

struct VramRow {
  double time_us = 0;
  uint64_t used_bytes = 0;
};

struct CpuSwitchRow {
  double timestamp_us = 0;
  int64_t cpu = 0;
  int64_t old_thread = 0;
  int64_t new_thread = 0;
  int64_t old_state = 0;
  int64_t wait_reason = 0;
};

struct CpuSampleRow {
  double timestamp_us = 0;
  int64_t cpu = 0;
  int64_t process = 0;
  int64_t thread = 0;
  std::string instruction_pointer;
  std::string stack;
};

struct EventState {
  uint64_t count = 0;
  uint64_t clamped = 0;
  bool json_enabled = true;
  PftraceWriter *pftrace = nullptr;
};

std::string Get(const CsvRow &row, const char *name) {
  const auto it = row.find(name);
  return it == row.end() ? std::string{} : it->second;
}

bool Exists(const fs::path &path) {
  std::error_code ec;
  return fs::exists(path, ec);
}

std::optional<double> ReadResumeTimestampUs(const Capture &capture) {
  const fs::path path = capture.Domain("launch");
  if (!Exists(path))
    return std::nullopt;
  CsvReader reader(path);
  CsvRow row;
  if (!reader.IsOpen() || !reader.Next(row))
    throw std::runtime_error(path.string() + ": cannot read launch metadata");
  const char *column =
      capture.IsLegacy() ? "Resume_Timestamp_ns" : "Resume_Timestamp";
  const uint64_t timestamp_ns = ToU64(Get(row, column));
  if (!timestamp_ns)
    throw std::runtime_error(path.string() + ": invalid resume timestamp");
  return static_cast<double>(timestamp_ns) / 1000.0;
}

Capture ResolveChromeCapture(const fs::path &source) {
  try {
    return ResolveCapture(source);
  } catch (const std::exception &) {
    const std::string raw = source.string();
    for (const std::string &suffix : {".api.csv", ".kernels.csv", ".vram.csv",
                                      ".roctx.csv", ".summary.txt"}) {
      if (!EndsWithInsensitive(raw, suffix))
        continue;
      const fs::path ops(raw.substr(0, raw.size() - suffix.size()));
      if (EndsWithInsensitive(ops.string(), "_ops.csv") && Exists(ops))
        return {fs::path(ops.string().substr(0, ops.string().size() - 8)),
                source, ops};
    }
    const fs::path csv(raw + ".csv");
    if (EndsWithInsensitive(csv.string(), "_ops.csv") && Exists(csv))
      return {fs::path(csv.string().substr(0, csv.string().size() - 8)), source,
              csv};
    throw;
  }
}

double Rounded(double value, int digits) {
  const double scale = std::pow(10.0, digits);
  return std::round(value * scale) / scale;
}

std::string SliceLabel(std::string value) {
  const std::string anonymous = "(anonymous namespace)::";
  for (size_t at = value.find(anonymous); at != std::string::npos;
       at = value.find(anonymous))
    value.erase(at, anonymous.size());

  int depth = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '<')
      ++depth;
    else if (value[i] == '>')
      --depth;
    else if (value[i] == '(' && depth == 0) {
      value.resize(i);
      break;
    }
  }
  depth = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '<')
      ++depth;
    else if (value[i] == '>')
      --depth;
    else if (value[i] == ' ' && depth == 0) {
      value.erase(0, i + 1);
      break;
    }
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())))
    value.erase(value.begin());
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  return value;
}

double TraceTimestamp(double absolute_us, double origin, EventState &state) {
  double value = Rounded(absolute_us - origin, 3);
  if (value < 0) {
    ++state.clamped;
    value = 0;
  }
  return value;
}

template <typename Fn>
void Emit(std::ofstream &out, EventState &state, Fn &&write) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> json(buffer);
  write(json);
  if (state.json_enabled) {
    if (state.count)
      out.put(',');
    out.write(buffer.GetString(),
              static_cast<std::streamsize>(buffer.GetSize()));
  }
  if (state.pftrace)
    state.pftrace->Event(buffer.GetString(), buffer.GetSize());
  ++state.count;
}

template <typename Writer>
void String(Writer &out, const char *key, const std::string &value) {
  out.Key(key);
  out.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

class CharStream {
public:
  explicit CharStream(const fs::path &path) : file_(path, std::ios::binary) {
    if (!file_)
      throw std::runtime_error(path.string() + ": cannot open JSON sidecar");
  }

  char Get() {
    if (pushed_) {
      const char ch = *pushed_;
      pushed_.reset();
      return ch;
    }
    if (at_ == size_) {
      file_.read(buffer_, sizeof(buffer_));
      size_ = static_cast<size_t>(file_.gcount());
      at_ = 0;
      if (!size_)
        return '\0';
    }
    return buffer_[at_++];
  }

  void Push(char ch) {
    if (ch)
      pushed_ = ch;
  }

  char Nonspace() {
    char ch = Get();
    while (ch && std::isspace(static_cast<unsigned char>(ch)))
      ch = Get();
    return ch;
  }

private:
  std::ifstream file_;
  char buffer_[65536]{};
  size_t at_ = 0;
  size_t size_ = 0;
  std::optional<char> pushed_;
};

std::string ReadJsonStringToken(CharStream &stream) {
  std::string token(1, '"');
  bool escaped = false;
  for (;;) {
    const char ch = stream.Get();
    if (!ch)
      throw std::runtime_error("unterminated JSON string");
    token.push_back(ch);
    if (escaped)
      escaped = false;
    else if (ch == '\\')
      escaped = true;
    else if (ch == '"')
      break;
  }
  rapidjson::Document value;
  value.Parse(token.c_str(), token.size());
  if (value.HasParseError() || !value.IsString())
    throw std::runtime_error("invalid JSON string token");
  return std::string(value.GetString(), value.GetStringLength());
}

std::string ReadRawJsonValue(CharStream &stream, char first) {
  if (!first)
    throw std::runtime_error("unexpected end of JSON");
  std::string token(1, first);
  if (first == '"') {
    bool escaped = false;
    for (;;) {
      const char ch = stream.Get();
      if (!ch)
        throw std::runtime_error("unterminated JSON string");
      token.push_back(ch);
      if (escaped)
        escaped = false;
      else if (ch == '\\')
        escaped = true;
      else if (ch == '"')
        return token;
    }
  }
  if (first == '[' || first == '{') {
    std::vector<char> stack(1, first);
    bool in_string = false;
    bool escaped = false;
    while (!stack.empty()) {
      const char ch = stream.Get();
      if (!ch)
        throw std::runtime_error("unterminated JSON value");
      token.push_back(ch);
      if (in_string) {
        if (escaped)
          escaped = false;
        else if (ch == '\\')
          escaped = true;
        else if (ch == '"')
          in_string = false;
      } else if (ch == '"')
        in_string = true;
      else if (ch == '[' || ch == '{')
        stack.push_back(ch);
      else if (ch == ']' || ch == '}') {
        const char expected = ch == ']' ? '[' : '{';
        if (stack.back() != expected)
          throw std::runtime_error("mismatched JSON delimiter");
        stack.pop_back();
      }
    }
    return token;
  }
  for (;;) {
    const char ch = stream.Get();
    if (!ch || std::isspace(static_cast<unsigned char>(ch)) || ch == ',' ||
        ch == ']' || ch == '}') {
      stream.Push(ch);
      return token;
    }
    token.push_back(ch);
  }
}

class NamedJsonCursor {
public:
  NamedJsonCursor(const fs::path &path,
                  std::initializer_list<const char *> names) {
    if (!Exists(path))
      return;
    stream_ = std::make_unique<CharStream>(path);
    std::unordered_set<std::string> wanted;
    for (const char *name : names)
      wanted.emplace(name);
    for (;;) {
      const char ch = stream_->Get();
      if (!ch)
        return;
      if (ch != '"')
        continue;
      const std::string key = ReadJsonStringToken(*stream_);
      if (!wanted.count(key))
        continue;
      if (stream_->Nonspace() != ':')
        continue;
      const char opening = stream_->Nonspace();
      if (opening == '[' || opening == '{') {
        opening_ = opening;
        active_ = true;
        return;
      }
      stream_->Push(opening);
    }
  }

  bool Next(rapidjson::Document &value, std::string &key) {
    if (!active_)
      return false;
    char ch = stream_->Nonspace();
    if (!first_) {
      if ((opening_ == '[' && ch == ']') || (opening_ == '{' && ch == '}')) {
        active_ = false;
        return false;
      }
      if (ch != ',')
        throw std::runtime_error("expected JSON container comma");
      ch = stream_->Nonspace();
    } else {
      first_ = false;
      if ((opening_ == '[' && ch == ']') || (opening_ == '{' && ch == '}')) {
        active_ = false;
        return false;
      }
    }

    key.clear();
    if (opening_ == '{') {
      if (ch != '"')
        throw std::runtime_error("expected JSON object key");
      key = ReadJsonStringToken(*stream_);
      if (stream_->Nonspace() != ':')
        throw std::runtime_error("expected JSON colon");
      ch = stream_->Nonspace();
    }
    const std::string raw = ReadRawJsonValue(*stream_, ch);
    value.Parse(raw.c_str(), raw.size());
    if (value.HasParseError()) {
      std::ostringstream message;
      message << "extension JSON value parse error at byte "
              << value.GetErrorOffset() << ": "
              << rapidjson::GetParseError_En(value.GetParseError());
      throw std::runtime_error(message.str());
    }
    return true;
  }

private:
  std::unique_ptr<CharStream> stream_;
  char opening_ = '\0';
  bool active_ = false;
  bool first_ = true;
};

const rapidjson::Value *First(const rapidjson::Value &row,
                              std::initializer_list<const char *> names) {
  if (!row.IsObject())
    return nullptr;
  for (const char *name : names) {
    const auto it = row.FindMember(name);
    if (it != row.MemberEnd() && !it->value.IsNull())
      return &it->value;
  }
  return nullptr;
}

std::string ValueText(const rapidjson::Value *value,
                      const std::string &fallback = {}) {
  if (!value || value->IsNull())
    return fallback;
  if (value->IsString())
    return std::string(value->GetString(), value->GetStringLength());
  if (value->IsBool())
    return value->GetBool() ? "True" : "False";
  if (value->IsInt64())
    return std::to_string(value->GetInt64());
  if (value->IsUint64())
    return std::to_string(value->GetUint64());
  if (value->IsDouble()) {
    std::ostringstream out;
    out << std::setprecision(15) << value->GetDouble();
    return out.str();
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value->Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}

uint64_t ValueU64(const rapidjson::Value *value, uint64_t fallback = 0) {
  if (!value || value->IsNull())
    return fallback;
  if (value->IsUint64())
    return value->GetUint64();
  if (value->IsInt64())
    return value->GetInt64() >= 0 ? static_cast<uint64_t>(value->GetInt64())
                                  : fallback;
  if (value->IsNumber())
    return static_cast<uint64_t>(value->GetDouble());
  return ToU64(ValueText(value), fallback);
}

int64_t ValueI64(const rapidjson::Value *value, int64_t fallback = 0) {
  if (!value || value->IsNull())
    return fallback;
  if (value->IsInt64())
    return value->GetInt64();
  if (value->IsUint64())
    return static_cast<int64_t>(value->GetUint64());
  if (value->IsNumber())
    return static_cast<int64_t>(value->GetDouble());
  return ToI64(ValueText(value), fallback);
}

double ValueDouble(const rapidjson::Value *value, double fallback = 0) {
  if (!value || value->IsNull())
    return fallback;
  if (value->IsNumber())
    return value->GetDouble();
  return ToDouble(ValueText(value), fallback);
}

uint64_t JsonCorr(const rapidjson::Value &row, const std::string &key) {
  const rapidjson::Value *value =
      First(row, {"Correlation_Id", "correlation_id", "Corr", "corr"});
  if (value) {
    if (value->IsObject())
      return ValueU64(First(*value, {"internal", "value", "handle"}));
    return ValueU64(value);
  }
  return ToU64(key);
}

std::string FormatArgs(const rapidjson::Value &value) {
  if (value.IsNull() || value.IsFalse())
    return {};
  if (value.IsString())
    return ValueText(&value);
  std::ostringstream out;
  bool first = true;
  auto append = [&](const std::string &text) {
    if (text.empty())
      return;
    if (!first)
      out << ' ';
    out << text;
    first = false;
  };
  if (value.IsObject()) {
    for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it)
      append(std::string(it->name.GetString(), it->name.GetStringLength()) +
             "=" + ValueText(&it->value));
    return out.str();
  }
  if (value.IsArray()) {
    for (const auto &item : value.GetArray()) {
      if (item.IsObject()) {
        const std::string name =
            ValueText(First(item, {"name", "Name", "key"}));
        if (!name.empty())
          append(name + "=" +
                 ValueText(First(item, {"value", "Value", "val"})));
      } else
        append(ValueText(&item));
    }
    return out.str();
  }
  return ValueText(&value);
}

std::string ArgsValue(const rapidjson::Value &row) {
  if (!row.IsObject())
    return FormatArgs(row);
  const rapidjson::Value *value =
      First(row, {"Args", "args", "decoded_args", "arguments"});
  if (!value)
    value = First(row, {"value"});
  return value ? FormatArgs(*value) : std::string{};
}

std::string MergeArgs(const std::string &left, const std::string &right) {
  if (left.empty())
    return right;
  if (right.empty())
    return left;
  std::set<std::string> names;
  std::istringstream old(left);
  std::string token;
  while (old >> token)
    names.insert(token.substr(0, token.find('=')));
  std::ostringstream out;
  out << left;
  std::istringstream extra(right);
  while (extra >> token) {
    if (names.insert(token.substr(0, token.find('='))).second)
      out << ' ' << token;
  }
  return out.str();
}

class ExtensionArgsCursor {
public:
  explicit ExtensionArgsCursor(const Capture &capture)
      : cursor_(capture.Domain("extensions"),
                {"api_args", "decoded_api_args", "api_arguments",
                 "hip_api_args", "apis"}) {}

  bool Next(uint64_t &corr, std::string &args) {
    rapidjson::Document row;
    std::string key;
    while (cursor_.Next(row, key)) {
      corr = JsonCorr(row, key);
      args = ArgsValue(row);
      if (corr && !args.empty())
        return true;
    }
    return false;
  }

private:
  NamedJsonCursor cursor_;
};

class CopyExtraCursor {
public:
  explicit CopyExtraCursor(const Capture &capture)
      : cursor_(capture.Domain("extensions"),
                {"memory_copy_extras", "copy_extras", "memory_copies",
                 "copy_records"}) {}

  bool Next(OpRow &row) {
    rapidjson::Document value;
    std::string key;
    while (cursor_.Next(value, key)) {
      if (!value.IsObject())
        continue;
      row = {};
      row.corr = JsonCorr(value, key);
      row.bytes = ValueU64(First(value, {"Bytes", "bytes", "size", "Size"}));
      if (!row.bytes) {
        const std::string args = ArgsValue(value);
        const std::string marker = "size=";
        for (size_t at = 0; at < args.size();) {
          const size_t end = args.find(' ', at);
          const std::string part =
              args.substr(at, end == std::string::npos ? end : end - at);
          if (part.rfind(marker, 0) == 0)
            row.bytes = ToU64(part.substr(marker.size()));
          if (end == std::string::npos)
            break;
          at = end + 1;
        }
      }
      row.device = ValueText(First(value, {"Device", "device"}));
      row.queue = ValueI64(First(value, {"Queue", "queue"}));
      row.stream = ValueI64(First(value, {"Stream", "stream"}));
      if (row.corr)
        return true;
    }
    return false;
  }

private:
  NamedJsonCursor cursor_;
};

class ExtensionOpCursor {
public:
  explicit ExtensionOpCursor(const Capture &capture)
      : cursor_(capture.Domain("extensions"),
                {"unmapped_gpu_commands", "gpu_operations",
                 "nonstandard_operations", "operations", "commands", "ops"}) {}

  bool Next(OpRow &out) {
    rapidjson::Document row;
    std::string key;
    while (cursor_.Next(row, key)) {
      if (!row.IsObject())
        continue;
      std::string op = Lower(ValueText(First(row, {"Op", "op", "operation"})));
      const std::string kind =
          ValueText(First(row, {"Kind", "kind", "direction"}));
      const std::string low_kind = Lower(kind);
      if (op.empty()) {
        if (low_kind.find("barrier") != std::string::npos)
          op = "barrier";
        else if (low_kind.find("fill") != std::string::npos ||
                 low_kind.find("memset") != std::string::npos)
          op = "copy";
      }
      if (op == "dispatch" ||
          (op == "copy" && low_kind.find("fill") == std::string::npos &&
           low_kind.find("memset") == std::string::npos))
        continue;
      if (op != "barrier" && op != "copy" && op != "other")
        continue;

      out = {};
      out.id = ValueText(First(row, {"Id", "id"}));
      out.op = op;
      out.kind = kind;
      out.bytes = ValueU64(First(row, {"Bytes", "bytes", "size"}));
      out.device =
          ValueText(First(row, {"Device", "device", "Agent_Id", "agent_id"}));
      out.queue = ValueI64(First(row, {"Queue", "queue", "Queue_Id"}));
      out.stream = ValueI64(First(row, {"Stream", "stream", "Stream_Id"}));
      out.thread = ValueI64(First(row, {"Thread", "thread", "Thread_Id"}));
      out.corr = JsonCorr(row, key);
      const rapidjson::Value *begin = First(row, {"Begin_us", "begin_us"});
      const rapidjson::Value *end = First(row, {"End_us", "end_us"});
      out.begin_us =
          begin ? ValueDouble(begin)
                : ValueI64(First(row, {"Start_Timestamp", "start_timestamp",
                                       "begin_ns"})) /
                      1000.0;
      out.end_us = end ? ValueDouble(end)
                       : ValueI64(First(row, {"End_Timestamp", "end_timestamp",
                                              "end_ns"})) /
                             1000.0;
      return true;
    }
    return false;
  }

private:
  NamedJsonCursor cursor_;
};

class KernelCsvCursor {
public:
  explicit KernelCsvCursor(const Capture &capture)
      : reader_(capture.Domain("kernels")) {}

  bool Next(OpRow &out) {
    CsvRow row;
    if (!reader_.IsOpen() || !reader_.Next(row))
      return false;
    const uint64_t start_ns = ToU64(Get(row, "Start_Timestamp"));
    const uint64_t end_ns = ToU64(Get(row, "End_Timestamp"));
    out = {};
    out.id = Get(row, "Dispatch_Id");
    out.op = "dispatch";
    out.kind = Get(row, "Kind");
    if (out.kind.empty())
      out.kind = "KERNEL_DISPATCH";
    out.kernel = Get(row, "Kernel_Name");
    out.device = Get(row, "Agent_Id");
    out.queue = ToI64(Get(row, "Queue_Id"));
    out.stream = ToI64(Get(row, "Stream_Id"));
    out.thread = ToI64(Get(row, "Thread_Id"));
    out.corr = ToU64(Get(row, "Correlation_Id"));
    out.begin_us = start_ns / 1000.0;
    out.end_us = end_ns / 1000.0;
    return true;
  }

private:
  CsvReader reader_;
};

std::unordered_set<std::string> GpuAgents(const Capture &capture) {
  std::unordered_set<std::string> result;
  CsvReader reader(capture.Domain("agents"));
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    if (Get(row, "Agent_Type") != "GPU")
      continue;
    result.insert("Agent " + std::to_string(ToI64(Get(row, "Node_Id"))));
    result.insert("Agent " +
                  std::to_string(ToI64(Get(row, "Logical_Node_Id"))));
  }
  return result;
}

class CopyCsvCursor {
public:
  explicit CopyCsvCursor(const Capture &capture)
      : reader_(capture.Domain("copy")), gpu_agents_(GpuAgents(capture)),
        extras_(capture) {
    has_extra_ = extras_.Next(extra_);
  }

  bool Next(OpRow &out) {
    CsvRow row;
    if (!reader_.IsOpen() || !reader_.Next(row))
      return false;
    const uint64_t start_ns = ToU64(Get(row, "Start_Timestamp"));
    const uint64_t end_ns = ToU64(Get(row, "End_Timestamp"));
    const std::string src = Get(row, "Source_Agent_Id");
    const std::string dst = Get(row, "Destination_Agent_Id");
    std::string device;
    if (gpu_agents_.count(dst))
      device = dst;
    else if (gpu_agents_.count(src))
      device = src;
    else
      device =
          (dst == "Agent 0" && !src.empty()) ? src : (!dst.empty() ? dst : src);

    out = {};
    out.op = "copy";
    out.kind = Get(row, "Direction");
    if (out.kind.empty())
      out.kind = "MEMORY_COPY";
    out.device = device;
    out.stream = ToI64(Get(row, "Stream_Id"));
    out.queue = out.stream;
    out.corr = ToU64(Get(row, "Correlation_Id"));
    out.begin_us = start_ns / 1000.0;
    out.end_us = end_ns / 1000.0;
    if (has_extra_ && extra_.corr == out.corr) {
      out.bytes = extra_.bytes;
      if (!extra_.device.empty())
        out.device = extra_.device;
      out.queue = extra_.queue;
      out.stream = extra_.stream;
      has_extra_ = extras_.Next(extra_);
    }
    return true;
  }

private:
  CsvReader reader_;
  std::unordered_set<std::string> gpu_agents_;
  CopyExtraCursor extras_;
  OpRow extra_;
  bool has_extra_ = false;
};

template <typename Fn> void ForEachOp(const Capture &capture, Fn &&fn) {
  if (capture.IsLegacy()) {
    CsvReader reader(capture.Domain("ops"));
    if (!reader.IsOpen())
      throw std::runtime_error(capture.Domain("ops").string() +
                               ": cannot open operations spool");
    CsvRow row;
    while (reader.Next(row)) {
      OpRow op;
      op.id = Get(row, "Id");
      op.op = Get(row, "Op");
      op.kind = Get(row, "Kind");
      op.kernel = Get(row, "Kernel_Name");
      op.bytes = ToU64(Get(row, "Bytes"));
      op.device = Get(row, "Device");
      op.queue = ToI64(Get(row, "Queue"));
      op.stream = ToI64(Get(row, "Stream"));
      op.thread = ToI64(Get(row, "Thread"));
      op.corr = ToU64(Get(row, "Corr"));
      op.begin_us = ToDouble(Get(row, "Begin_us"));
      op.end_us = ToDouble(Get(row, "End_us"));
      fn(op);
    }
    return;
  }

  KernelCsvCursor kernels(capture);
  CopyCsvCursor copies(capture);
  ExtensionOpCursor extensions(capture);
  OpRow kernel, copy, extension;
  bool have_kernel = kernels.Next(kernel);
  bool have_copy = copies.Next(copy);
  bool have_extension = extensions.Next(extension);
  while (have_kernel || have_copy || have_extension) {
    OpRow *next = nullptr;
    int stream = -1;
    if (have_kernel) {
      next = &kernel;
      stream = 0;
    }
    if (have_copy && (!next || copy.begin_us < next->begin_us)) {
      next = &copy;
      stream = 1;
    }
    if (have_extension && (!next || extension.begin_us < next->begin_us)) {
      next = &extension;
      stream = 2;
    }
    fn(*next);
    if (stream == 0)
      have_kernel = kernels.Next(kernel);
    else if (stream == 1)
      have_copy = copies.Next(copy);
    else
      have_extension = extensions.Next(extension);
  }
}

LanePlan PlanLanes(const std::vector<OpTiming> &timings, double from,
                   double to) {
  LanePlan plan;
  plan.by_op.assign(timings.size(), std::numeric_limits<uint32_t>::max());

  std::vector<size_t> order;
  order.reserve(timings.size());
  for (size_t i = 0; i < timings.size(); ++i)
    if (timings[i].end_us >= from && timings[i].begin_us <= to)
      order.push_back(i);

  std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    const OpTiming &a = timings[left];
    const OpTiming &b = timings[right];
    if (a.queue != b.queue)
      return a.queue < b.queue;
    if (a.begin_us != b.begin_us)
      return a.begin_us < b.begin_us;
    if (a.end_us != b.end_us)
      return a.end_us < b.end_us;
    return left < right;
  });

  using ActiveLane = std::pair<double, uint32_t>;
  std::priority_queue<ActiveLane, std::vector<ActiveLane>,
                      std::greater<ActiveLane>>
      active;
  std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>
      available;
  std::optional<int64_t> current_queue;
  uint32_t next_lane = 0;
  for (const size_t index : order) {
    const OpTiming &timing = timings[index];
    if (!current_queue || timing.queue != *current_queue) {
      current_queue = timing.queue;
      active = {};
      available = {};
      next_lane = 0;
    }
    while (!active.empty() && active.top().first <= timing.begin_us) {
      available.push(active.top().second);
      active.pop();
    }

    uint32_t lane = 0;
    if (available.empty())
      lane = next_lane++;
    else {
      lane = available.top();
      available.pop();
    }
    plan.by_op[index] = lane;
    plan.counts[timing.queue] = std::max(plan.counts[timing.queue], lane + 1);
    active.emplace(std::max(timing.begin_us, timing.end_us), lane);
  }
  return plan;
}

template <typename Fn>
void ForEachApi(const Capture &capture, bool include_args, Fn &&fn) {
  CsvReader reader(capture.Domain("api"));
  if (!reader.IsOpen())
    return;

  std::unique_ptr<ExtensionArgsCursor> extras;
  uint64_t extra_corr = 0;
  std::string extra_args;
  bool have_extra = false;
  if (!capture.IsLegacy() && include_args) {
    extras = std::make_unique<ExtensionArgsCursor>(capture);
    have_extra = extras->Next(extra_corr, extra_args);
  }

  CsvRow row;
  while (reader.Next(row)) {
    ApiRow api;
    if (capture.IsLegacy()) {
      api.corr = ToU64(Get(row, "Corr"));
      api.api = Get(row, "Api");
      api.thread = ToI64(Get(row, "Thread"));
      api.enter_us = ToDouble(Get(row, "Enter_us"));
      api.exit_us = ToDouble(Get(row, "Exit_us"));
      if (include_args)
        api.args = Get(row, "Args");
    } else {
      api.corr = ToU64(Get(row, "Correlation_Id"));
      api.api = Get(row, "Function");
      api.thread = ToI64(Get(row, "Thread_Id"));
      const uint64_t start_ns = ToU64(Get(row, "Start_Timestamp"));
      const uint64_t end_ns = ToU64(Get(row, "End_Timestamp"));
      api.enter_us = start_ns / 1000.0;
      api.exit_us = end_ns / 1000.0;
      if (have_extra && extra_corr == api.corr) {
        api.args = MergeArgs(api.args, extra_args);
        have_extra = extras->Next(extra_corr, extra_args);
      }
    }
    fn(api);
  }
}

template <typename Fn> void ForEachRoctx(const Capture &capture, Fn &&fn) {
  CsvReader reader(capture.Domain("roctx"));
  if (!reader.IsOpen())
    return;

  CsvRow row;
  while (reader.Next(row)) {
    RoctxRow event;
    if (capture.IsLegacy()) {
      event.operation = Get(row, "Op");
      event.process = ToI64(Get(row, "Process"));
      event.thread = ToI64(Get(row, "Thread"));
      event.range_id = ToU64(Get(row, "Range_Id"));
      event.timestamp_us = ToDouble(Get(row, "Timestamp_us"));
      event.message = Get(row, "Message");
    } else {
      event.operation = Get(row, "Operation");
      event.process = ToI64(Get(row, "Process_Id"));
      event.thread = ToI64(Get(row, "Thread_Id"));
      event.range_id = ToU64(Get(row, "Range_Id"));
      event.timestamp_us = ToU64(Get(row, "Timestamp")) / 1000.0;
      event.message = Get(row, "Message");
    }
    fn(event);
  }
}

std::unordered_map<std::string, KernelAttrs>
ReadKernelInfo(const Capture &capture) {
  std::unordered_map<std::string, KernelAttrs> result;
  if (capture.IsLegacy()) {
    CsvReader reader(capture.Domain("kernels"));
    CsvRow row;
    while (reader.IsOpen() && reader.Next(row)) {
      result[Get(row, "Kernel_Name")] = {
          Get(row, "Theoretical_Occupancy_Pct"), Get(row, "VGPR"),
          Get(row, "Scratch_Bytes"), Get(row, "Static_LDS")};
    }
    return result;
  }

  struct Extra {
    std::string name;
    uint64_t id = 0;
    KernelAttrs attrs;
  };
  std::unordered_map<std::string, KernelAttrs> by_name;
  std::unordered_map<uint64_t, KernelAttrs> by_id;
  NamedJsonCursor cursor(
      capture.Domain("extensions"),
      {"kernel_properties", "kernel_extras", "kernels", "kernel_info"});
  rapidjson::Document value;
  std::string key;
  while (cursor.Next(value, key)) {
    if (!value.IsObject())
      continue;
    KernelAttrs attrs{
        ValueText(First(value, {"Theoretical_Occupancy_Pct",
                                "theoretical_occupancy_pct", "occupancy_pct"})),
        ValueText(First(value, {"VGPR", "vgpr", "VGPR_Count"})),
        ValueText(
            First(value, {"Scratch_Bytes", "scratch_bytes", "Scratch_Size"})),
        ValueText(
            First(value, {"Static_LDS", "static_lds", "LDS_Block_Size"}))};
    const std::string name =
        ValueText(First(value, {"Kernel_Name", "kernel_name", "name"}));
    const uint64_t id =
        ValueU64(First(value, {"Kernel_Id", "kernel_id", "id"}), ToU64(key));
    if (!name.empty())
      by_name[name] = attrs;
    if (id)
      by_id[id] = attrs;
  }

  CsvReader reader(capture.Domain("kernels"));
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    const std::string name = Get(row, "Kernel_Name");
    if (result.count(name))
      continue;
    KernelAttrs attrs{"", Get(row, "VGPR_Count"), Get(row, "Scratch_Size"),
                      Get(row, "LDS_Block_Size")};
    const auto named = by_name.find(name);
    const auto numbered = by_id.find(ToU64(Get(row, "Kernel_Id")));
    if (named != by_name.end())
      attrs = named->second;
    else if (numbered != by_id.end())
      attrs = numbered->second;
    if (attrs.vgpr.empty())
      attrs.vgpr = Get(row, "VGPR_Count");
    if (attrs.scratch.empty())
      attrs.scratch = Get(row, "Scratch_Size");
    if (attrs.static_lds.empty())
      attrs.static_lds = Get(row, "LDS_Block_Size");
    result.emplace(name, std::move(attrs));
  }
  return result;
}

template <typename Fn> void ForEachVram(const Capture &capture, Fn &&fn) {
  if (capture.IsLegacy()) {
    CsvReader reader(capture.Domain("vram"));
    CsvRow row;
    while (reader.IsOpen() && reader.Next(row))
      fn(VramRow{ToDouble(Get(row, "Time_us")), ToU64(Get(row, "Used_Bytes"))});
    return;
  }
  NamedJsonCursor cursor(
      capture.Domain("extensions"),
      {"vram_samples", "vram", "device_memory_samples", "memory_samples"});
  rapidjson::Document row;
  std::string key;
  while (cursor.Next(row, key)) {
    if (!row.IsObject())
      continue;
    const rapidjson::Value *time = First(row, {"Time_us", "time_us"});
    const double time_us =
        time ? ValueDouble(time)
             : ValueI64(First(row, {"Timestamp", "timestamp", "timestamp_ns",
                                    "time_ns"})) /
                   1000.0;
    const uint64_t total =
        ValueU64(First(row, {"Total_Bytes", "total_bytes", "total"}));
    const uint64_t free =
        ValueU64(First(row, {"Free_Bytes", "free_bytes", "free"}));
    const rapidjson::Value *used =
        First(row, {"Used_Bytes", "used_bytes", "used"});
    fn(VramRow{time_us,
               used ? ValueU64(used) : (total >= free ? total - free : 0)});
  }
}

template <typename Fn> void ForEachCpuSwitch(const Capture &capture, Fn &&fn) {
  CsvReader reader(capture.Domain("cpu_sched"));
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    fn(CpuSwitchRow{
        ToDouble(Get(row, "Timestamp_ns")) / 1000.0,
        ToI64(Get(row, "Cpu")),
        ToI64(Get(row, "Old_Thread")),
        ToI64(Get(row, "New_Thread")),
        ToI64(Get(row, "Old_State")),
        ToI64(Get(row, "Wait_Reason")),
    });
  }
}

template <typename Fn> void ForEachCpuSample(const Capture &capture, Fn &&fn) {
  CsvReader reader(capture.Domain("cpu_samples"));
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    fn(CpuSampleRow{
        ToDouble(Get(row, "Timestamp_ns")) / 1000.0,
        ToI64(Get(row, "Cpu")),
        ToI64(Get(row, "Process_Id")),
        ToI64(Get(row, "Thread_Id")),
        Get(row, "Instruction_Pointer"),
        Get(row, "Stack"),
    });
  }
}

std::string DeviceName(const Capture &capture) {
  if (capture.IsLegacy()) {
    std::ifstream file(capture.Domain("summary"));
    std::string line;
    while (std::getline(file, line)) {
      if (line.rfind("device ", 0) != 0)
        continue;
      const size_t colon = line.find(':');
      if (colon == std::string::npos)
        continue;
      std::string name = line.substr(colon + 1);
      while (!name.empty() &&
             std::isspace(static_cast<unsigned char>(name.front())))
        name.erase(name.begin());
      while (!name.empty() &&
             std::isspace(static_cast<unsigned char>(name.back())))
        name.pop_back();
      return name;
    }
    return {};
  }
  CsvReader reader(capture.Domain("agents"));
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    if (Get(row, "Agent_Type") != "GPU")
      continue;
    for (const char *key : {"Product_Name", "Name", "Model_Name"}) {
      const std::string value = Get(row, key);
      if (!value.empty())
        return value;
    }
  }
  return {};
}

double ParseNumber(const std::string &text, const std::string &option) {
  char *end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end)
    throw std::runtime_error(option + ": invalid number '" + text + "'");
  return value;
}

int64_t ParseIndex(const std::string &text) {
  size_t used = 0;
  try {
    const int64_t value = std::stoll(text, &used, 10);
    if (used == text.size())
      return value;
  } catch (const std::exception &) {
  }
  throw std::runtime_error("--phase: invalid run index '" + text + "'");
}

Options ParseOptions(const std::vector<std::string> &args) {
  Options options;
  std::vector<std::string> positional;
  bool positional_only = false;
  auto value = [&](size_t &i, const std::string &option) -> std::string {
    if (++i >= args.size())
      throw std::runtime_error(option + " requires a value");
    return args[i];
  };
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];
    if (positional_only)
      positional.push_back(arg);
    else if (arg == "--")
      positional_only = true;
    else if (arg == "--all")
      options.all = true;
    else if (arg == "--no-flows")
      options.no_flows = true;
    else if (arg == "--flows")
      options.force_flows = true;
    else if (arg == "--chrome-json") {
      options.output = value(i, arg);
      options.json_requested = true;
      options.format_requested = true;
    } else if (arg == "--chrome-pftrace") {
      options.pftrace_output = value(i, arg);
      options.pftrace_requested = true;
      options.format_requested = true;
    } else if (arg == "--quiet")
      options.quiet = true;
    else if (arg == "--start-ms")
      options.start_ms = ParseNumber(value(i, arg), arg);
    else if (arg.rfind("--start-ms=", 0) == 0)
      options.start_ms = ParseNumber(arg.substr(11), "--start-ms");
    else if (arg == "--span-ms")
      options.span_ms = ParseNumber(value(i, arg), arg);
    else if (arg.rfind("--span-ms=", 0) == 0)
      options.span_ms = ParseNumber(arg.substr(10), "--span-ms");
    else if (arg == "--phase")
      options.phase = value(i, arg);
    else if (arg.rfind("--phase=", 0) == 0)
      options.phase = arg.substr(8);
    else if (arg == "-h" || arg == "--help") {
      std::cout << "usage: ktrace_postprocess.exe chrome CAPTURE [out_json]\n"
                << "       [--start-ms N] [--span-ms N] [--all]\n"
                << "       [--phase NAME[:N]] [--flows|--no-flows] [--quiet]\n"
                << "       [--chrome-json PATH] [--chrome-pftrace PATH]\n"
                << "  --flows  with --all, keep every host API and correlation "
                   "arrow\n";
      return {};
    } else if (!arg.empty() && arg[0] == '-')
      throw std::runtime_error("unknown chrome option '" + arg + "'");
    else
      positional.push_back(arg);
  }
  if (positional.empty())
    throw std::runtime_error(
        "chrome requires a capture prefix or legacy *_ops.csv");
  if (positional.size() > 2)
    throw std::runtime_error(
        "chrome accepts only CAPTURE and optional out_json");
  options.source = positional[0];
  if (positional.size() == 2) {
    options.output = positional[1];
    options.json_requested = true;
  }
  if (!options.format_requested && positional.size() == 1)
    options.json_requested = true;
  if (options.force_flows && options.no_flows)
    throw std::runtime_error("--flows and --no-flows are mutually exclusive");
  if (options.all && !options.force_flows)
    options.no_flows = true;
  return options;
}

std::string ScopeText(bool all, double from, double to, double origin) {
  if (all)
    return "full capture";
  std::ostringstream out;
  out << std::fixed << std::setprecision(1) << (from - origin) / 1000.0 << ".."
      << (to - origin) / 1000.0 << " ms";
  return out.str();
}

void WriteProcessName(std::ofstream &out, EventState &state, int pid,
                      const std::string &name) {
  Emit(out, state, [&](auto &json) {
    json.StartObject();
    String(json, "ph", "M");
    json.Key("pid");
    json.Int(pid);
    String(json, "name", "process_name");
    json.Key("args");
    json.StartObject();
    String(json, "name", name);
    json.EndObject();
    json.EndObject();
  });
  Emit(out, state, [&](auto &json) {
    json.StartObject();
    String(json, "ph", "M");
    json.Key("pid");
    json.Int(pid);
    String(json, "name", "process_sort_index");
    json.Key("args");
    json.StartObject();
    json.Key("sort_index");
    json.Int(pid);
    json.EndObject();
    json.EndObject();
  });
}

void WriteThreadName(std::ofstream &out, EventState &state, int pid,
                     int64_t tid, const std::string &name) {
  Emit(out, state, [&](auto &json) {
    json.StartObject();
    String(json, "ph", "M");
    json.Key("pid");
    json.Int(pid);
    json.Key("tid");
    json.Int64(tid);
    String(json, "name", "thread_name");
    json.Key("args");
    json.StartObject();
    String(json, "name", name);
    json.EndObject();
    json.EndObject();
  });
}

std::vector<std::pair<std::string, std::string>>
SplitApiArgs(const std::string &api, const std::string &text) {
  std::vector<std::pair<std::string, std::string>> result{{"api", api}};
  std::istringstream input(text);
  std::string token;
  while (input >> token) {
    const size_t equal = token.find('=');
    const std::string key = token.substr(0, equal);
    const std::string value =
        equal == std::string::npos ? std::string{} : token.substr(equal + 1);
    auto it = std::find_if(result.begin(), result.end(),
                           [&](const auto &item) { return item.first == key; });
    if (it == result.end())
      result.emplace_back(key, value);
    else
      it->second = value;
  }
  return result;
}
} // namespace

int ChromeCommand(const std::vector<std::string> &args) {
  const Options options = ParseOptions(args);
  if (options.source.empty())
    return 0;
  const Capture capture = ResolveChromeCapture(options.source);
  const fs::path json_output = options.output.empty()
                                   ? capture.AutoOutput("_ktrace_chrome.json")
                                   : options.output;
  const fs::path pftrace_output =
      options.pftrace_output.empty()
          ? capture.AutoOutput("_ktrace_perfetto.pftrace")
          : options.pftrace_output;

  std::optional<double> gpu_lo;
  std::vector<std::string> raw_names;
  std::unordered_set<std::string> raw_seen;
  std::vector<std::pair<double, std::string>> marks;
  std::vector<OpTiming> op_timings;
  ForEachOp(capture, [&](const OpRow &row) {
    op_timings.push_back({row.queue, row.begin_us, row.end_us});
    if (!gpu_lo || row.begin_us < *gpu_lo)
      gpu_lo = row.begin_us;
    if (row.op != "dispatch" || row.kernel.empty())
      return;
    if (raw_seen.insert(row.kernel).second)
      raw_names.push_back(row.kernel);
    for (const char *word : {"prefill", "decode"}) {
      if (row.kernel.find(word) != std::string::npos) {
        marks.emplace_back(row.begin_us, word);
        break;
      }
    }
  });
  const std::vector<Phase> phases = SegmentPhases(std::move(marks), 0.5);

  std::optional<double> host_lo;
  std::unordered_set<int64_t> target_threads;
  ForEachApi(capture, false, [&](const ApiRow &row) {
    if (!host_lo || row.enter_us < *host_lo)
      host_lo = row.enter_us;
    target_threads.insert(row.thread);
  });
  std::optional<double> roctx_lo;
  std::vector<RoctxRow> roctx_events;
  ForEachRoctx(capture, [&](const RoctxRow &row) {
    roctx_events.push_back(row);
    target_threads.insert(row.thread);
    if (!roctx_lo || row.timestamp_us < *roctx_lo)
      roctx_lo = row.timestamp_us;
  });
  std::sort(roctx_events.begin(), roctx_events.end(),
            [](const RoctxRow &left, const RoctxRow &right) {
              return left.timestamp_us < right.timestamp_us;
            });
  const std::optional<double> resume_origin = ReadResumeTimestampUs(capture);
  double origin = resume_origin.value_or(
      gpu_lo.value_or(host_lo.value_or(roctx_lo.value_or(0.0))));
  if (!resume_origin) {
    if (host_lo)
      origin = std::min(origin, *host_lo);
    // Preserve the historical GPU/HIP time origin. ROCTx annotations are
    // projected onto that timeline and must not shift existing slices.
    if (!gpu_lo && !host_lo && roctx_lo)
      origin = *roctx_lo;
  }

  double from = 0;
  double to = 0;
  if (options.all) {
    from = -std::numeric_limits<double>::infinity();
    to = std::numeric_limits<double>::infinity();
  } else if (!options.phase.empty()) {
    const size_t colon = options.phase.find(':');
    const std::string name = options.phase.substr(0, colon);
    std::vector<const Phase *> runs;
    std::set<std::string> found;
    for (const auto &phase : phases) {
      found.insert(phase.tag);
      if (phase.tag == name)
        runs.push_back(&phase);
    }
    if (runs.empty()) {
      std::ostringstream message;
      message << "no phase named '" << name << "'; this capture has: ";
      if (found.empty())
        message << "none";
      else {
        bool first = true;
        for (const auto &item : found) {
          if (!first)
            message << ", ";
          message << item;
          first = false;
        }
      }
      throw std::runtime_error(message.str());
    }
    int64_t index = static_cast<int64_t>(runs.size()) - 1;
    if (colon != std::string::npos && colon + 1 < options.phase.size())
      index = ParseIndex(options.phase.substr(colon + 1));
    if (index < 0)
      index += static_cast<int64_t>(runs.size());
    if (index < 0 || static_cast<size_t>(index) >= runs.size())
      throw std::runtime_error("phase run index is out of range");
    from = runs[static_cast<size_t>(index)]->begin_us;
    to = std::min(runs[static_cast<size_t>(index)]->end_us,
                  from + options.span_ms * 1000.0);
  } else {
    from = origin + options.start_ms.value_or(0.0) * 1000.0;
    to = from + options.span_ms * 1000.0;
  }

  const auto kernel_info = ReadKernelInfo(capture);
  auto labels = ResolveKernelNames(raw_names);
  for (auto &[raw, label] : labels)
    label = SliceLabel(std::move(label));
  const std::string scope = ScopeText(options.all, from, to, origin);
  const LanePlan lane_plan = PlanLanes(op_timings, from, to);

  std::vector<char> output_buffer(1024 * 1024);
  std::ofstream out;
  if (options.json_requested) {
    out.rdbuf()->pubsetbuf(output_buffer.data(),
                           static_cast<std::streamsize>(output_buffer.size()));
    out.open(json_output, std::ios::binary);
    if (!out)
      throw std::runtime_error(json_output.string() + ": cannot create output");
    out << "{\"displayTimeUnit\":\"ms\",\"otherData\":";
  }
  std::unique_ptr<PftraceWriter> pftrace;
  if (options.pftrace_requested)
    pftrace = std::make_unique<PftraceWriter>(
        pftrace_output,
        capture.IsLegacy() ? fs::path{} : capture.Domain("cpu_images"), true);
  {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> json(buffer);
    json.StartObject();
    String(json, "producer",
           "ktrace (hipRegisterTracerCallback, native runtime timestamps)");
    String(json, "format", "Chrome Trace Event JSON");
    String(
        json, "source",
        AbsoluteUtf8(capture.IsLegacy() ? capture.legacy_ops : capture.prefix));
    String(json, "captured", NowIso8601());
    String(json, "scope", scope);
    String(json, "host", Hostname());
    String(json, "time_origin",
           resume_origin ? "ResumeThread QPC" : "first captured event");
    const std::string device = DeviceName(capture);
    if (!device.empty())
      String(json, "device", device);
    if (!phases.empty()) {
      std::vector<std::string> order;
      std::unordered_map<std::string, size_t> counts;
      for (const auto &phase : phases) {
        if (!counts.count(phase.tag))
          order.push_back(phase.tag);
        ++counts[phase.tag];
      }
      std::ostringstream summary;
      for (size_t i = 0; i < order.size(); ++i) {
        if (i)
          summary << ", ";
        summary << order[i] << " x" << counts[order[i]];
      }
      String(json, "phases", summary.str());
    }
    json.EndObject();
    if (options.json_requested)
      out.write(buffer.GetString(),
                static_cast<std::streamsize>(buffer.GetSize()));
    if (pftrace)
      pftrace->Metadata(std::string(buffer.GetString(), buffer.GetSize()));
  }
  if (options.json_requested)
    out << ",\"traceEvents\":[";

  EventState state;
  state.json_enabled = options.json_requested;
  state.pftrace = pftrace.get();
  WriteProcessName(out, state, kGpuPid, "GPU (hardware queues)");
  WriteProcessName(out, state, kHostPid, "Host (HIP API)");
  if (!roctx_events.empty())
    WriteProcessName(out, state, kRoctxPid, "ROCTx annotations");
  const bool have_cpu_sched = Exists(capture.Domain("cpu_sched"));
  const bool have_cpu_samples = Exists(capture.Domain("cpu_samples"));
  if (have_cpu_sched)
    WriteProcessName(out, state, kCpuPid, "CPU scheduling");
  if (have_cpu_samples)
    WriteProcessName(out, state, kCpuSamplePid, "CPU samples");

  uint64_t roctx_count = 0;
  std::unordered_set<int64_t> roctx_threads_seen;
  std::unordered_map<int64_t, std::vector<RoctxRow>> nested_ranges;
  std::unordered_map<uint64_t, RoctxRow> process_ranges;
  auto write_roctx_range = [&](const RoctxRow &begin, const RoctxRow &end,
                               bool cross_thread) {
    if (end.timestamp_us < from || begin.timestamp_us > to)
      return;
    const double lo = std::max(begin.timestamp_us, from);
    const double hi = std::min(end.timestamp_us, to);
    if (hi < lo)
      return;
    if (roctx_threads_seen.insert(begin.thread).second)
      WriteThreadName(out, state, kRoctxPid, begin.thread,
                      "ROCTx thread " + std::to_string(begin.thread));
    Emit(out, state, [&](auto &json) {
      json.StartObject();
      String(json, "ph", "X");
      json.Key("pid");
      json.Int(kRoctxPid);
      json.Key("tid");
      json.Int64(begin.thread);
      String(json, "name",
             begin.message.empty() ? "ROCTx range" : begin.message);
      String(json, "cat", "roctx");
      json.Key("ts");
      json.Double(TraceTimestamp(lo, origin, state));
      json.Key("dur");
      json.Double(Rounded(hi - lo, 3));
      json.Key("args");
      json.StartObject();
      json.Key("range_id");
      json.Uint64(begin.range_id);
      json.Key("cross_thread");
      json.Bool(cross_thread);
      if (cross_thread) {
        json.Key("stop_thread");
        json.Int64(end.thread);
      }
      json.EndObject();
      json.EndObject();
    });
    ++roctx_count;
  };

  for (const RoctxRow &event : roctx_events) {
    if (event.operation == "MARK") {
      if (event.timestamp_us < from || event.timestamp_us > to)
        continue;
      if (roctx_threads_seen.insert(event.thread).second)
        WriteThreadName(out, state, kRoctxPid, event.thread,
                        "ROCTx thread " + std::to_string(event.thread));
      Emit(out, state, [&](auto &json) {
        json.StartObject();
        String(json, "ph", "i");
        String(json, "s", "t");
        json.Key("pid");
        json.Int(kRoctxPid);
        json.Key("tid");
        json.Int64(event.thread);
        String(json, "name",
               event.message.empty() ? "ROCTx mark" : event.message);
        String(json, "cat", "roctx");
        json.Key("ts");
        json.Double(TraceTimestamp(event.timestamp_us, origin, state));
        json.EndObject();
      });
      ++roctx_count;
    } else if (event.operation == "RANGE_PUSH") {
      nested_ranges[event.thread].push_back(event);
    } else if (event.operation == "RANGE_POP") {
      auto &stack = nested_ranges[event.thread];
      if (!stack.empty()) {
        const RoctxRow begin = stack.back();
        stack.pop_back();
        write_roctx_range(begin, event, false);
      }
    } else if (event.operation == "RANGE_START") {
      process_ranges[event.range_id] = event;
    } else if (event.operation == "RANGE_STOP") {
      const auto begin = process_ranges.find(event.range_id);
      if (begin != process_ranges.end()) {
        write_roctx_range(begin->second, event, true);
        process_ranges.erase(begin);
      }
    }
  }

  out.flush();
  if (!options.quiet)
    std::cout << "[scan] GPU command domains ...\n";

  uint64_t phase_count = 0;
  if (!phases.empty()) {
    WriteProcessName(out, state, kPhasePid,
                     "Phases (inferred from kernel names)");
    WriteThreadName(out, state, kPhasePid, 0, "phase");
    for (size_t i = 0; i < phases.size(); ++i) {
      const Phase &phase = phases[i];
      if (phase.end_us < from || phase.begin_us > to)
        continue;
      const double lo = std::max(phase.begin_us, from);
      const double hi = std::min(phase.end_us, to);
      Emit(out, state, [&](auto &json) {
        json.StartObject();
        String(json, "ph", "X");
        json.Key("pid");
        json.Int(kPhasePid);
        json.Key("tid");
        json.Int(0);
        String(json, "name", phase.tag);
        String(json, "cat", "phase");
        json.Key("ts");
        json.Double(TraceTimestamp(lo, origin, state));
        json.Key("dur");
        json.Double(Rounded(hi - lo, 3));
        json.Key("args");
        json.StartObject();
        json.Key("run");
        json.Uint64(i);
        json.Key("full_ms");
        json.Double(Rounded((phase.end_us - phase.begin_us) / 1000.0, 1));
        json.Key("clipped");
        json.Bool(lo > phase.begin_us || hi < phase.end_us);
        json.EndObject();
        json.EndObject();
      });
      ++phase_count;
    }
  }

  std::unordered_set<int64_t> queues_seen;
  std::unordered_set<uint64_t> corr_gpu;
  uint64_t gpu_count = 0;
  size_t op_index = 0;
  ForEachOp(capture, [&](const OpRow &row) {
    const size_t index = op_index++;
    if (row.end_us < from || row.begin_us > to)
      return;
    const uint32_t lane = lane_plan.by_op.at(index);
    if (lane == std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("internal lane assignment mismatch");
    const int64_t tid = row.queue * 1000 + static_cast<int64_t>(lane);
    const std::string queue = row.device + "/" + std::to_string(row.queue);
    if (queues_seen.insert(tid).second) {
      std::ostringstream label;
      label << "queue " << queue;
      if (lane)
        label << " ~" << lane;
      WriteThreadName(out, state, kGpuPid, tid, label.str());
    }

    std::string name = row.kind;
    if (row.op == "dispatch") {
      const auto label = labels.find(row.kernel);
      name = label == labels.end() ? row.kernel : label->second;
    }
    if (name.empty())
      name = row.op;
    Emit(out, state, [&](auto &json) {
      json.StartObject();
      String(json, "ph", "X");
      json.Key("pid");
      json.Int(kGpuPid);
      json.Key("tid");
      json.Int64(tid);
      String(json, "name", name);
      String(json, "cat", row.op);
      json.Key("ts");
      json.Double(TraceTimestamp(row.begin_us, origin, state));
      json.Key("dur");
      json.Double(Rounded(row.end_us - row.begin_us, 3));
      json.Key("args");
      json.StartObject();
      String(json, "op", row.op);
      String(json, "kind", row.kind);
      String(json, "queue", queue);
      if (row.op == "dispatch") {
        String(json, "mangled", row.kernel);
        const auto info = kernel_info.find(row.kernel);
        if (info != kernel_info.end()) {
          String(json, "occupancy_pct", info->second.occupancy);
          String(json, "VGPR", info->second.vgpr);
          String(json, "scratch_bytes", info->second.scratch);
          String(json, "static_lds", info->second.static_lds);
        }
      } else {
        json.Key("bytes");
        json.Uint64(row.bytes);
      }
      json.EndObject();
      json.EndObject();
    });
    ++gpu_count;

    if (row.corr && !options.no_flows) {
      corr_gpu.insert(row.corr);
      Emit(out, state, [&](auto &json) {
        json.StartObject();
        String(json, "ph", "f");
        String(json, "bp", "e");
        json.Key("pid");
        json.Int(kGpuPid);
        json.Key("tid");
        json.Int64(tid);
        json.Key("id");
        json.Uint64(row.corr);
        String(json, "cat", "launch");
        String(json, "name", "issued by");
        json.Key("ts");
        json.Double(TraceTimestamp(row.begin_us, origin, state));
        json.EndObject();
      });
    }
  });

  if (!options.quiet)
    std::cout << "[scan] HIP API domain ...\n";
  uint64_t host_count = 0;
  std::unordered_set<int64_t> threads_seen;
  ForEachApi(capture, true, [&](const ApiRow &row) {
    if (row.exit_us < from || row.enter_us > to)
      return;
    if (options.all && !options.force_flows && !kBlockingApis.count(row.api))
      return;
    if (threads_seen.insert(row.thread).second)
      WriteThreadName(out, state, kHostPid, row.thread,
                      "thread " + std::to_string(row.thread));
    const auto api_args = SplitApiArgs(row.api, row.args);
    Emit(out, state, [&](auto &json) {
      json.StartObject();
      String(json, "ph", "X");
      json.Key("pid");
      json.Int(kHostPid);
      json.Key("tid");
      json.Int64(row.thread);
      String(json, "name", row.api);
      String(json, "cat", "hip_api");
      json.Key("ts");
      json.Double(TraceTimestamp(row.enter_us, origin, state));
      json.Key("dur");
      json.Double(Rounded(row.exit_us - row.enter_us, 3));
      json.Key("args");
      json.StartObject();
      for (const auto &[key, value] : api_args)
        String(json, key.c_str(), value);
      json.EndObject();
      json.EndObject();
    });
    ++host_count;

    if (row.corr && !options.no_flows && corr_gpu.count(row.corr)) {
      Emit(out, state, [&](auto &json) {
        json.StartObject();
        String(json, "ph", "s");
        String(json, "bp", "e");
        json.Key("pid");
        json.Int(kHostPid);
        json.Key("tid");
        json.Int64(row.thread);
        json.Key("id");
        json.Uint64(row.corr);
        String(json, "cat", "launch");
        String(json, "name", "issued by");
        json.Key("ts");
        json.Double(TraceTimestamp(row.enter_us, origin, state));
        json.EndObject();
      });
    }
  });

  uint64_t cpu_sched_count = 0;
  if (have_cpu_sched) {
    std::unordered_map<int64_t, double> cpu_start;
    std::unordered_set<int64_t> cpu_tracks;
    ForEachCpuSwitch(capture, [&](const CpuSwitchRow &row) {
      const auto previous = cpu_start.find(row.cpu);
      if (previous != cpu_start.end() && target_threads.count(row.old_thread) &&
          row.timestamp_us >= previous->second) {
        const double lo = std::max(previous->second, from);
        const double hi = std::min(row.timestamp_us, to);
        if (hi >= lo) {
          if (cpu_tracks.insert(row.cpu).second)
            WriteThreadName(out, state, kCpuPid, row.cpu,
                            "CPU " + std::to_string(row.cpu));
          Emit(out, state, [&](auto &json) {
            json.StartObject();
            String(json, "ph", "X");
            json.Key("pid");
            json.Int(kCpuPid);
            json.Key("tid");
            json.Int64(row.cpu);
            String(json, "name", "thread " + std::to_string(row.old_thread));
            String(json, "cat", "cpu_sched");
            json.Key("ts");
            json.Double(TraceTimestamp(lo, origin, state));
            json.Key("dur");
            json.Double(Rounded(hi - lo, 3));
            json.Key("args");
            json.StartObject();
            json.Key("thread_id");
            json.Int64(row.old_thread);
            json.Key("state");
            json.Int64(row.old_state);
            json.Key("wait_reason");
            json.Int64(row.wait_reason);
            json.EndObject();
            json.EndObject();
          });
          ++cpu_sched_count;
        }
      }
      cpu_start[row.cpu] = row.timestamp_us;
    });
  }

  uint64_t cpu_sample_count = 0;
  if (have_cpu_samples) {
    std::unordered_set<int64_t> sample_tracks;
    ForEachCpuSample(capture, [&](const CpuSampleRow &row) {
      if (!target_threads.count(row.thread) || row.timestamp_us < from ||
          row.timestamp_us > to)
        return;
      if (sample_tracks.insert(row.thread).second)
        WriteThreadName(out, state, kCpuSamplePid, row.thread,
                        "thread " + std::to_string(row.thread));
      Emit(out, state, [&](auto &json) {
        json.StartObject();
        String(json, "ph", "i");
        String(json, "s", "t");
        json.Key("pid");
        json.Int(kCpuSamplePid);
        json.Key("tid");
        json.Int64(row.thread);
        String(json, "name", "CPU sample");
        String(json, "cat", "cpu_sample");
        json.Key("ts");
        json.Double(TraceTimestamp(row.timestamp_us, origin, state));
        json.Key("args");
        json.StartObject();
        String(json, "instruction_pointer", row.instruction_pointer);
        String(json, "stack", row.stack);
        json.Key("cpu");
        json.Int64(row.cpu);
        json.Key("process_id");
        json.Int64(row.process);
        json.EndObject();
        json.EndObject();
      });
      ++cpu_sample_count;
    });
  }

  uint64_t vram_count = 0;
  ForEachVram(capture, [&](const VramRow &row) {
    if (row.time_us < from || row.time_us > to)
      return;
    Emit(out, state, [&](auto &json) {
      json.StartObject();
      String(json, "ph", "C");
      json.Key("pid");
      json.Int(kGpuPid);
      String(json, "name", "device memory");
      json.Key("ts");
      json.Double(TraceTimestamp(row.time_us, origin, state));
      json.Key("args");
      json.StartObject();
      json.Key("used_MB");
      json.Double(Rounded(row.used_bytes / 1048576.0, 1));
      json.EndObject();
      json.EndObject();
    });
    ++vram_count;
  });
  if (options.json_requested) {
    out << "]}";
    out.close();
    if (!out)
      throw std::runtime_error(json_output.string() + ": write failed");
  }
  if (pftrace)
    pftrace->Close();

  uintmax_t json_size = 0;
  uintmax_t pftrace_size = 0;
  std::error_code ec;
  if (options.json_requested)
    json_size = fs::file_size(json_output, ec);
  if (options.pftrace_requested)
    pftrace_size = fs::file_size(pftrace_output, ec);
  size_t max_lanes = 0;
  for (const auto &[queue, lanes] : lane_plan.counts)
    max_lanes = std::max(max_lanes, static_cast<size_t>(lanes));
  if (!options.quiet) {
    std::cout << "scope       : " << scope << '\n'
              << "phases      : " << phase_count << " band(s) of "
              << phases.size() << " in the capture\n"
              << "GPU slices  : " << gpu_count << " on " << queues_seen.size()
              << " tracks (" << lane_plan.counts.size() << " queues, "
              << max_lanes << " lanes at most)\n"
              << "host slices : " << host_count << " on " << threads_seen.size()
              << " threads"
              << (options.all && !options.force_flows
                      ? "  (blocking calls only)"
                      : "")
              << '\n'
              << "ROCTx events: " << roctx_count << '\n'
              << "CPU sched   : " << cpu_sched_count << '\n'
              << "CPU samples : " << cpu_sample_count << '\n'
              << "vram points : " << vram_count << '\n'
              << "events      : " << state.count << '\n';
    if (state.clamped)
      std::cout << "[warn] " << state.clamped
                << " events had a timestamp before the trace origin and were "
                   "clamped to 0\n";
    if (options.json_requested)
      std::cout << "-> " << json_output.string() << "  (" << std::fixed
                << std::setprecision(1) << json_size / 1e6 << " MB)\n";
    if (options.pftrace_requested)
      std::cout << "-> " << pftrace_output.string() << "  (" << std::fixed
                << std::setprecision(1) << pftrace_size / 1e6 << " MB, "
                << pftrace->PacketCount() << " packets)\n";
    std::cout << "   open at https://ui.perfetto.dev\n";
  }
  return 0;
}
} // namespace ktrace
