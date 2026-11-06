/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ktrace {
namespace {
constexpr double kLowOccupancyPct = 50.0;
constexpr size_t kMaxCorrelationQueues = 400000;

std::string Get(const CsvRow &row, const char *key) {
  const auto it = row.find(key);
  return it == row.end() ? std::string{} : it->second;
}

std::string Commas(uint64_t value) {
  std::string text = std::to_string(value);
  for (ptrdiff_t at = static_cast<ptrdiff_t>(text.size()) - 3; at > 0; at -= 3)
    text.insert(static_cast<size_t>(at), 1, ',');
  return text;
}

std::string Fixed(double value, int precision) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string Left(std::string value, size_t width) {
  if (value.size() < width)
    value.append(width - value.size(), ' ');
  return value;
}

std::string Right(const std::string &value, size_t width) {
  return value.size() >= width ? value
                               : std::string(width - value.size(), ' ') + value;
}

std::string MergeArgs(const std::string &left, const std::string &right) {
  if (left.empty())
    return right;
  if (right.empty())
    return left;
  std::set<std::string> names;
  std::istringstream in(left);
  std::string token;
  while (in >> token)
    names.insert(token.substr(0, token.find('=')));
  std::string out = left;
  std::istringstream extra(right);
  while (extra >> token) {
    if (names.insert(token.substr(0, token.find('='))).second)
      out += " " + token;
  }
  return out;
}

std::string ArgValue(const std::string &args, const std::string &name) {
  const std::string marker = name + "=";
  std::istringstream in(args);
  std::string token;
  while (in >> token)
    if (token.rfind(marker, 0) == 0)
      return token.substr(marker.size());
  return {};
}

struct CountTime {
  uint64_t count = 0;
  double us = 0;
};

struct Transfer {
  uint64_t count = 0;
  uint64_t bytes = 0;
  double us = 0;
};

struct OpRow {
  std::string op;
  std::string kind;
  std::string kernel;
  std::string device;
  std::string queue;
  uint64_t bytes = 0;
  uint64_t corr = 0;
  double begin_us = 0;
  double end_us = 0;
};

struct OpsAggregate {
  std::unordered_map<std::string, double> kernel_time;
  std::unordered_map<std::string, uint64_t> kernel_count;
  std::unordered_map<std::string, double> kernel_max;
  std::unordered_map<std::string, CountTime> queues;
  std::unordered_map<std::string, Transfer> transfers;
  std::unordered_map<uint64_t, std::string> corr_queue;
  std::unordered_map<std::string, uint64_t> counts;
  std::unordered_map<std::string, std::unordered_map<std::string, double>>
      phase_kernel;
  std::unordered_map<std::string, double> phase_busy;
  std::vector<Phase> phases;
  std::vector<std::pair<double, std::string>> phase_marks;
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  double busy = 0;

  void Add(const OpRow &row) {
    const double duration = row.end_us - row.begin_us;
    ++counts[row.op];
    busy += duration;
    lo = std::min(lo, row.begin_us);
    hi = std::max(hi, row.end_us);

    const std::string queue = row.device + "/" + row.queue;
    auto &q = queues[queue];
    ++q.count;
    q.us += duration;

    if (row.op == "dispatch") {
      kernel_time[row.kernel] += duration;
      ++kernel_count[row.kernel];
      kernel_max[row.kernel] = std::max(kernel_max[row.kernel], duration);
      if (row.corr && corr_queue.size() < kMaxCorrelationQueues)
        corr_queue[row.corr] = queue;
      const std::string lower = Lower(row.kernel);
      const char *phase = lower.find("prefill") != std::string::npos ? "prefill"
                          : lower.find("decode") != std::string::npos ? "decode"
                                                                      : nullptr;
      if (phase)
        phase_marks.emplace_back(row.begin_us, phase);
    } else if (row.op == "copy") {
      auto &transfer = transfers[row.kind];
      ++transfer.count;
      transfer.bytes += row.bytes;
      transfer.us += duration;
    }
  }

  double Span() const {
    return std::isfinite(lo) && std::isfinite(hi) ? hi - lo : 0;
  }

  void FinishPhases() {
    phases = SegmentPhases(std::move(phase_marks), 0.5);
    phase_marks.clear();
    phase_kernel.clear();
    phase_busy.clear();
  }

  void AddPhase(const OpRow &row) {
    if (phases.empty())
      return;
    const double midpoint = (row.begin_us + row.end_us) * 0.5;
    const auto after = std::upper_bound(
        phases.begin(), phases.end(), midpoint,
        [](double time, const Phase &phase) { return time < phase.begin_us; });
    if (after == phases.begin())
      return;
    const Phase &phase = *std::prev(after);
    if (midpoint > phase.end_us)
      return;
    const double duration = row.end_us - row.begin_us;
    phase_busy[phase.tag] += duration;
    if (row.op == "dispatch")
      phase_kernel[phase.tag][row.kernel] += duration;
  }
};

struct KernelData {
  std::string name;
  uint64_t id = 0;
  std::string vgpr;
  std::string scratch;
  std::string lds;
  std::string occupancy;
};

struct ApiAggregate {
  std::unordered_map<std::string, CountTime> per_api;
  std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>>
      stream_queue;
  std::unordered_map<std::string, CountTime> stream_wait;
  std::unordered_map<std::string, uint64_t> allocations;
  std::unordered_map<std::string, double> phase_wait;
  uint64_t alloc_peak = 0;
  uint64_t alloc_current = 0;
  uint64_t alloc_total = 0;
  uint64_t unmatched_free = 0;
  uint64_t total_calls = 0;
};

struct VramAggregate {
  uint64_t count = 0;
  uint64_t minimum = 0;
  uint64_t maximum = 0;
  uint64_t first = 0;
  uint64_t last = 0;
  double first_time = 0;
  double last_time = 0;

  void Add(double time_us, uint64_t used) {
    if (!count) {
      minimum = maximum = first = used;
      first_time = time_us;
    }
    minimum = std::min(minimum, used);
    maximum = std::max(maximum, used);
    last = used;
    last_time = time_us;
    ++count;
  }
};

struct RoctxRangeStats {
  uint64_t count = 0;
  double total_us = 0;
  double max_us = 0;
  uint64_t cross_thread = 0;
};

struct RoctxAggregate {
  uint64_t events = 0;
  uint64_t marks = 0;
  uint64_t unmatched = 0;
  std::unordered_map<std::string, RoctxRangeStats> ranges;
};

struct RoctxPoint {
  std::string message;
  int64_t thread = 0;
  uint64_t range_id = 0;
  double timestamp_us = 0;
};

RoctxAggregate ReadRoctx(const Capture &capture) {
  RoctxAggregate result;
  CsvReader reader(capture.Domain("roctx"));
  if (!reader.IsOpen())
    return result;

  std::unordered_map<int64_t, std::vector<RoctxPoint>> stacks;
  std::unordered_map<uint64_t, RoctxPoint> process_ranges;
  CsvRow row;
  while (reader.Next(row)) {
    const bool legacy = capture.IsLegacy();
    const std::string operation = Get(row, legacy ? "Op" : "Operation");
    RoctxPoint point{
        Get(row, "Message"),
        ToI64(Get(row, legacy ? "Thread" : "Thread_Id")),
        ToU64(Get(row, "Range_Id")),
        legacy ? ToDouble(Get(row, "Timestamp_us"))
               : ToU64(Get(row, "Timestamp")) / 1000.0,
    };
    ++result.events;

    auto add_range = [&](const RoctxPoint &begin, bool cross_thread) {
      const double duration =
          std::max(0.0, point.timestamp_us - begin.timestamp_us);
      auto &range =
          result.ranges[begin.message.empty() ? std::string{"<unnamed>"}
                                              : begin.message];
      ++range.count;
      range.total_us += duration;
      range.max_us = std::max(range.max_us, duration);
      if (cross_thread)
        ++range.cross_thread;
    };

    if (operation == "MARK") {
      ++result.marks;
    } else if (operation == "RANGE_PUSH") {
      stacks[point.thread].push_back(point);
    } else if (operation == "RANGE_POP") {
      auto &stack = stacks[point.thread];
      if (stack.empty())
        ++result.unmatched;
      else {
        const RoctxPoint begin = stack.back();
        stack.pop_back();
        add_range(begin, false);
      }
    } else if (operation == "RANGE_START") {
      process_ranges[point.range_id] = point;
    } else if (operation == "RANGE_STOP") {
      const auto begin = process_ranges.find(point.range_id);
      if (begin == process_ranges.end())
        ++result.unmatched;
      else {
        add_range(begin->second, true);
        process_ranges.erase(begin);
      }
    }
  }
  for (const auto &[_, stack] : stacks)
    result.unmatched += stack.size();
  result.unmatched += process_ranges.size();
  return result;
}

std::string ReadDeviceName(const Capture &capture) {
  if (!capture.IsLegacy()) {
    CsvReader reader(capture.Domain("agents"));
    CsvRow row;
    while (reader.IsOpen() && reader.Next(row)) {
      if (Get(row, "Agent_Type") != "GPU")
        continue;
      for (const char *field : {"Model_Name", "Product_Name", "Name"}) {
        const std::string value = Get(row, field);
        if (!value.empty())
          return value;
      }
    }
    return {};
  }

  std::ifstream file(capture.Domain("summary"));
  std::string line;
  while (std::getline(file, line)) {
    if (line.rfind("device", 0) != 0)
      continue;
    const size_t colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    size_t begin = colon + 1;
    while (begin < line.size() &&
           std::isspace(static_cast<unsigned char>(line[begin])))
      ++begin;
    return line.substr(begin);
  }
  return {};
}

struct RuntimeInfo {
  std::string path;
  std::string version;
};

// The runtime table is written once per capture directory, with one row per
// traced process, so a PID-prefixed capture finds no table of its own and has
// to fall back to the unprefixed one. Picking the first row there would report
// another process's runtime, so match on this capture's own process id, which
// its launch metadata carries.
RuntimeInfo ReadRuntimeInfo(const Capture &capture) {
  uint64_t pid = 0;
  {
    CsvReader launch(capture.Domain("launch"));
    CsvRow row;
    if (launch.IsOpen() && launch.Next(row))
      pid = ToU64(Get(row, "Process_Id"));
  }

  fs::path table = capture.Domain("runtime");
  if (!fs::exists(table)) {
    // Strip a trailing _<pid> from the prefix to reach the shared table.
    const std::string name = capture.prefix.filename().u8string();
    const size_t cut = name.find_last_of('_');
    if (cut != std::string::npos &&
        name.find_first_not_of("0123456789", cut + 1) == std::string::npos) {
      const Capture base{capture.prefix.parent_path() /
                             fs::u8path(name.substr(0, cut)),
                         {},
                         {}};
      table = base.Domain("runtime");
    }
  }

  CsvReader reader(table);
  if (!reader.IsOpen())
    return {};
  CsvRow row;
  RuntimeInfo first;
  while (reader.Next(row)) {
    RuntimeInfo entry{Get(row, "Hip_Runtime_Path"), Get(row, "File_Version")};
    if (first.path.empty())
      first = entry;
    if (pid && ToU64(Get(row, "Process_Id")) == pid)
      return entry;
  }
  return first;
}

OpRow LegacyOp(const CsvRow &row) {
  return {Get(row, "Op"),
          Get(row, "Kind"),
          Get(row, "Kernel_Name"),
          Get(row, "Device"),
          Get(row, "Queue"),
          ToU64(Get(row, "Bytes")),
          ToU64(Get(row, "Corr")),
          ToDouble(Get(row, "Begin_us")),
          ToDouble(Get(row, "End_us"))};
}

void ReadLegacyOps(const fs::path &path, OpsAggregate &out) {
  CsvReader reader(path);
  if (!reader.IsOpen())
    throw std::runtime_error("cannot open " + path.string());
  CsvRow row;
  while (reader.Next(row))
    out.Add(LegacyOp(row));
}

void ReadLegacyPhaseOps(const fs::path &path, OpsAggregate &out) {
  CsvReader reader(path);
  if (!reader.IsOpen())
    throw std::runtime_error("cannot open " + path.string());
  CsvRow row;
  while (reader.Next(row))
    out.AddPhase(LegacyOp(row));
}

void ReadKernelOps(const fs::path &path, OpsAggregate &out,
                   std::unordered_map<std::string, KernelData> &kernel_rows) {
  if (!fs::exists(path))
    return;
  CsvReader reader(path);
  if (!reader.IsOpen())
    throw std::runtime_error("cannot open " + path.string());
  CsvRow row;
  while (reader.Next(row)) {
    const std::string name = Get(row, "Kernel_Name");
    if (!name.empty() && !kernel_rows.count(name))
      kernel_rows[name] = {name,
                           ToU64(Get(row, "Kernel_Id")),
                           Get(row, "VGPR_Count"),
                           Get(row, "Scratch_Size"),
                           Get(row, "LDS_Block_Size"),
                           ""};
    const uint64_t start = ToU64(Get(row, "Start_Timestamp"));
    const uint64_t end = ToU64(Get(row, "End_Timestamp"));
    out.Add({"dispatch", Get(row, "Kind"), name, Get(row, "Agent_Id"),
             Get(row, "Queue_Id"), 0, ToU64(Get(row, "Correlation_Id")),
             start / 1000.0, end / 1000.0});
  }
}

void ReadKernelPhaseOps(const fs::path &path, OpsAggregate &out) {
  if (!fs::exists(path))
    return;
  CsvReader reader(path);
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    const uint64_t start = ToU64(Get(row, "Start_Timestamp"));
    const uint64_t end = ToU64(Get(row, "End_Timestamp"));
    out.AddPhase({"dispatch", Get(row, "Kind"), Get(row, "Kernel_Name"),
                  Get(row, "Agent_Id"), Get(row, "Queue_Id"), 0,
                  ToU64(Get(row, "Correlation_Id")), start / 1000.0,
                  end / 1000.0});
  }
}

void ReadCopyPhaseOps(const fs::path &path, OpsAggregate &out) {
  if (!fs::exists(path))
    return;
  CsvReader reader(path);
  CsvRow row;
  while (reader.IsOpen() && reader.Next(row)) {
    const uint64_t start = ToU64(Get(row, "Start_Timestamp"));
    const uint64_t end = ToU64(Get(row, "End_Timestamp"));
    out.AddPhase({"copy", Get(row, "Direction"), "", "", "", 0,
                  ToU64(Get(row, "Correlation_Id")), start / 1000.0,
                  end / 1000.0});
  }
}

std::unordered_set<std::string> ReadGpuAgents(const fs::path &path) {
  std::unordered_set<std::string> result;
  if (!fs::exists(path))
    return result;
  CsvReader reader(path);
  CsvRow row;
  while (reader.Next(row)) {
    if (Get(row, "Agent_Type") != "GPU")
      continue;
    result.insert("Agent " + std::to_string(ToU64(Get(row, "Node_Id"))));
    result.insert("Agent " +
                  std::to_string(ToU64(Get(row, "Logical_Node_Id"))));
  }
  return result;
}

struct CopyExtra {
  uint64_t bytes = 0;
  std::string device;
  std::string queue;
};

class CopyJoin {
public:
  CopyJoin(const Capture &capture, OpsAggregate &aggregate)
      : aggregate_(aggregate),
        gpu_agents_(ReadGpuAgents(capture.Domain("agents"))),
        reader_(capture.Domain("copy")) {
    open_ = reader_.IsOpen();
    if (open_)
      has_row_ = reader_.Next(row_);
  }

  void Supplement(uint64_t corr, const CopyExtra &extra) {
    while (has_row_ && ToU64(Get(row_, "Correlation_Id")) < corr)
      Advance({});
    if (has_row_ && ToU64(Get(row_, "Correlation_Id")) == corr)
      Advance(extra);
  }

  void Finish() {
    while (has_row_)
      Advance({});
  }

private:
  void Advance(const CopyExtra &extra) {
    const uint64_t start = ToU64(Get(row_, "Start_Timestamp"));
    const uint64_t end = ToU64(Get(row_, "End_Timestamp"));
    const std::string src = Get(row_, "Source_Agent_Id");
    const std::string dst = Get(row_, "Destination_Agent_Id");
    std::string device = extra.device;
    if (device.empty()) {
      if (gpu_agents_.count(dst))
        device = dst;
      else if (gpu_agents_.count(src))
        device = src;
      else
        device =
            dst == "Agent 0" && !src.empty() ? src : (!dst.empty() ? dst : src);
    }
    std::string queue =
        extra.queue.empty() ? Get(row_, "Stream_Id") : extra.queue;
    aggregate_.Add({"copy", Get(row_, "Direction"), "", device, queue,
                    extra.bytes, ToU64(Get(row_, "Correlation_Id")),
                    start / 1000.0, end / 1000.0});
    has_row_ = reader_.Next(row_);
  }

  OpsAggregate &aggregate_;
  std::unordered_set<std::string> gpu_agents_;
  CsvReader reader_;
  CsvRow row_;
  bool open_ = false;
  bool has_row_ = false;
};

void ApplyAllocation(const std::string &operation, const std::string &ptr,
                     uint64_t size, ApiAggregate &out) {
  const std::string upper = Lower(operation);
  if (upper.find("allocate") != std::string::npos &&
      upper.find("free") == std::string::npos) {
    const auto old = out.allocations.find(ptr);
    if (old != out.allocations.end())
      out.alloc_current -= old->second;
    out.allocations[ptr] = size;
    out.alloc_current += size;
    out.alloc_total += size;
    out.alloc_peak = std::max(out.alloc_peak, out.alloc_current);
  } else if (upper.find("free") != std::string::npos) {
    const auto it = out.allocations.find(ptr);
    if (it == out.allocations.end())
      ++out.unmatched_free;
    else {
      out.alloc_current -= it->second;
      out.allocations.erase(it);
    }
  }
}

void ProcessApiRow(const CsvRow &row, const std::string &supplied_args,
                   bool legacy, bool allocation_domain, const OpsAggregate &ops,
                   ApiAggregate &out) {
  const std::string api = legacy ? Get(row, "Api") : Get(row, "Function");
  const uint64_t corr =
      ToU64(legacy ? Get(row, "Corr") : Get(row, "Correlation_Id"));
  const double enter = legacy ? ToDouble(Get(row, "Enter_us"))
                              : ToU64(Get(row, "Start_Timestamp")) / 1000.0;
  const double host = legacy ? ToDouble(Get(row, "Host_us"))
                             : (ToU64(Get(row, "End_Timestamp")) -
                                ToU64(Get(row, "Start_Timestamp"))) /
                                   1000.0;
  auto &item = out.per_api[api];
  ++item.count;
  item.us += host;
  ++out.total_calls;

  const std::string args =
      MergeArgs(legacy ? Get(row, "Args") : "", supplied_args);
  const std::string stream = ArgValue(args, "stream");
  if (!stream.empty()) {
    if (api == "hipStreamSynchronize") {
      auto &wait = out.stream_wait[stream];
      ++wait.count;
      wait.us += host;
      for (const auto &phase : ops.phases) {
        if (phase.begin_us <= enter && enter <= phase.end_us) {
          out.phase_wait[phase.tag] += host;
          break;
        }
      }
    } else {
      const auto queue = ops.corr_queue.find(corr);
      if (queue != ops.corr_queue.end())
        ++out.stream_queue[stream][queue->second];
    }
  }

  if (!allocation_domain && (api == "hipMalloc" || api == "hipHostMalloc")) {
    const std::string ptr = ArgValue(args, "ptr");
    const std::string size = ArgValue(args, "size");
    if (!ptr.empty() && !size.empty())
      ApplyAllocation("allocate", ptr, ToU64(size), out);
  } else if (!allocation_domain && (api == "hipFree" || api == "hipHostFree")) {
    const std::string ptr = ArgValue(args, "ptr");
    if (!ptr.empty())
      ApplyAllocation("free", ptr, 0, out);
  }
}

class ApiJoin {
public:
  ApiJoin(const Capture &capture, const OpsAggregate &ops,
          ApiAggregate &aggregate)
      : ops_(ops), aggregate_(aggregate),
        allocation_domain_(fs::exists(capture.Domain("alloc"))),
        reader_(capture.Domain("api")) {
    open_ = reader_.IsOpen();
    if (open_)
      has_row_ = reader_.Next(row_);
  }

  bool IsOpen() const { return open_; }

  void Supplement(uint64_t corr, const std::string &args) {
    while (has_row_ && ToU64(Get(row_, "Correlation_Id")) < corr)
      Advance({});
    if (has_row_ && ToU64(Get(row_, "Correlation_Id")) == corr)
      Advance(args);
  }

  void Finish() {
    while (has_row_)
      Advance({});
  }

private:
  void Advance(const std::string &args) {
    ProcessApiRow(row_, args, false, allocation_domain_, ops_, aggregate_);
    has_row_ = reader_.Next(row_);
  }

  const OpsAggregate &ops_;
  ApiAggregate &aggregate_;
  bool allocation_domain_ = false;
  CsvReader reader_;
  CsvRow row_;
  bool open_ = false;
  bool has_row_ = false;
};

void ReadLegacyApi(const Capture &capture, const OpsAggregate &ops,
                   ApiAggregate &out) {
  CsvReader reader(capture.Domain("api"));
  if (!reader.IsOpen())
    return;
  CsvRow row;
  while (reader.Next(row))
    ProcessApiRow(row, {}, true, false, ops, out);
}

void ReadAllocations(const fs::path &path, ApiAggregate &out) {
  if (!fs::exists(path))
    return;
  CsvReader reader(path);
  CsvRow row;
  while (reader.Next(row))
    ApplyAllocation(Get(row, "Operation"), Get(row, "Address"),
                    ToU64(Get(row, "Allocation_Size")), out);
}

void ReadLegacyVram(const fs::path &path, VramAggregate &out) {
  CsvReader reader(path);
  if (!reader.IsOpen())
    return;
  CsvRow row;
  while (reader.Next(row))
    out.Add(ToDouble(Get(row, "Time_us")), ToU64(Get(row, "Used_Bytes")));
}

struct JsonNode {
  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Object,
    Array
  } type = Type::Null;
  std::string scalar;
  std::vector<std::pair<std::string, JsonNode>> object;
  std::vector<JsonNode> array;

  const JsonNode *Find(std::initializer_list<const char *> names) const {
    if (type != Type::Object)
      return nullptr;
    for (const char *name : names)
      for (const auto &entry : object)
        if (entry.first == name)
          return &entry.second;
    return nullptr;
  }

  std::string Text() const {
    if (type == Type::String || type == Type::Number || type == Type::Bool)
      return scalar;
    return {};
  }

  uint64_t Integer(uint64_t fallback = 0) const {
    if (type == Type::Object) {
      const JsonNode *internal =
          Find({"internal", "Internal", "handle", "Handle"});
      return internal ? internal->Integer(fallback) : fallback;
    }
    return ToU64(Text(), fallback);
  }

  double Number(double fallback = 0) const {
    return ToDouble(Text(), fallback);
  }
};

class NodeBuilder {
public:
  void Reset() {
    root_ = {};
    stack_.clear();
    keys_.clear();
  }
  JsonNode Take() { return std::move(root_); }

  void Null() { Add({}); }
  void Bool(bool value) {
    JsonNode node;
    node.type = JsonNode::Type::Bool;
    node.scalar = value ? "true" : "false";
    Add(std::move(node));
  }
  void Number(const std::string &value) {
    JsonNode node;
    node.type = JsonNode::Type::Number;
    node.scalar = value;
    Add(std::move(node));
  }
  void String(const char *value, rapidjson::SizeType length) {
    JsonNode node;
    node.type = JsonNode::Type::String;
    node.scalar.assign(value, length);
    Add(std::move(node));
  }
  void Key(const char *value, rapidjson::SizeType length) {
    if (!keys_.empty())
      keys_.back().assign(value, length);
  }
  void StartObject() { Start(JsonNode::Type::Object); }
  void StartArray() { Start(JsonNode::Type::Array); }
  void End() {
    if (!stack_.empty())
      stack_.pop_back();
    if (!keys_.empty())
      keys_.pop_back();
  }

private:
  JsonNode *Add(JsonNode node) {
    if (stack_.empty()) {
      root_ = std::move(node);
      return &root_;
    }
    JsonNode *parent = stack_.back();
    if (parent->type == JsonNode::Type::Array) {
      parent->array.push_back(std::move(node));
      return &parent->array.back();
    }
    parent->object.emplace_back(keys_.back(), std::move(node));
    keys_.back().clear();
    return &parent->object.back().second;
  }
  void Start(JsonNode::Type type) {
    JsonNode node;
    node.type = type;
    JsonNode *added = Add(std::move(node));
    stack_.push_back(added);
    keys_.push_back({});
  }

  JsonNode root_;
  std::vector<JsonNode *> stack_;
  std::vector<std::string> keys_;
};

std::string NodeText(const JsonNode &node,
                     std::initializer_list<const char *> names,
                     const std::string &fallback = {}) {
  const JsonNode *value = node.Find(names);
  return value ? value->Text() : fallback;
}

uint64_t NodeInt(const JsonNode &node,
                 std::initializer_list<const char *> names,
                 uint64_t fallback = 0) {
  const JsonNode *value = node.Find(names);
  return value ? value->Integer(fallback) : fallback;
}

double NodeNumber(const JsonNode &node,
                  std::initializer_list<const char *> names,
                  double fallback = 0) {
  const JsonNode *value = node.Find(names);
  return value ? value->Number(fallback) : fallback;
}

std::string FormatArgs(const JsonNode &value) {
  if (value.type == JsonNode::Type::String ||
      value.type == JsonNode::Type::Number ||
      value.type == JsonNode::Type::Bool)
    return value.Text();
  std::ostringstream out;
  bool first = true;
  if (value.type == JsonNode::Type::Object) {
    for (const auto &[key, item] : value.object) {
      if (!first)
        out << ' ';
      out << key << '=' << item.Text();
      first = false;
    }
  } else if (value.type == JsonNode::Type::Array) {
    for (const auto &item : value.array) {
      const std::string name = NodeText(item, {"name", "Name", "key"});
      const std::string text = NodeText(item, {"value", "Value", "val"});
      if (name.empty())
        continue;
      if (!first)
        out << ' ';
      out << name << '=' << text;
      first = false;
    }
  }
  return out.str();
}

enum class ExtensionSection { None, Api, Copy, Ops, Kernels, Vram };

ExtensionSection ClassifySection(const std::string &name) {
  static const std::unordered_map<std::string, ExtensionSection> names = {
      {"api_args", ExtensionSection::Api},
      {"decoded_api_args", ExtensionSection::Api},
      {"api_arguments", ExtensionSection::Api},
      {"hip_api_args", ExtensionSection::Api},
      {"apis", ExtensionSection::Api},
      {"memory_copy_extras", ExtensionSection::Copy},
      {"copy_extras", ExtensionSection::Copy},
      {"memory_copies", ExtensionSection::Copy},
      {"copy_records", ExtensionSection::Copy},
      {"unmapped_gpu_commands", ExtensionSection::Ops},
      {"gpu_operations", ExtensionSection::Ops},
      {"nonstandard_operations", ExtensionSection::Ops},
      {"operations", ExtensionSection::Ops},
      {"commands", ExtensionSection::Ops},
      {"ops", ExtensionSection::Ops},
      {"kernel_properties", ExtensionSection::Kernels},
      {"kernel_extras", ExtensionSection::Kernels},
      {"kernels", ExtensionSection::Kernels},
      {"kernel_info", ExtensionSection::Kernels},
      {"vram_samples", ExtensionSection::Vram},
      {"vram", ExtensionSection::Vram},
      {"device_memory_samples", ExtensionSection::Vram},
      {"memory_samples", ExtensionSection::Vram},
  };
  const auto it = names.find(name);
  return it == names.end() ? ExtensionSection::None : it->second;
}

class ExtensionConsumer {
public:
  ExtensionConsumer(OpsAggregate &ops, ApiJoin &api, CopyJoin &copy,
                    VramAggregate &vram)
      : ops_(ops), api_(api), copy_(copy), vram_(vram) {}

  void Item(ExtensionSection section, const std::string &key,
            const JsonNode &value) {
    const JsonNode *row = &value;
    if (value.type != JsonNode::Type::Object &&
        section != ExtensionSection::Api)
      return;
    if (section == ExtensionSection::Api) {
      const uint64_t corr =
          value.type == JsonNode::Type::Object
              ? NodeInt(value,
                        {"Correlation_Id", "correlation_id", "Corr", "corr"},
                        ToU64(key))
              : ToU64(key);
      const JsonNode *args_node =
          value.type == JsonNode::Type::Object
              ? value.Find(
                    {"Args", "args", "decoded_args", "arguments", "value"})
              : &value;
      if (corr && args_node)
        api_.Supplement(corr, FormatArgs(*args_node));
      return;
    }
    if (section == ExtensionSection::Copy) {
      const uint64_t corr =
          NodeInt(*row, {"Correlation_Id", "correlation_id", "Corr", "corr"},
                  ToU64(key));
      CopyExtra extra;
      extra.bytes = NodeInt(*row, {"Bytes", "bytes", "size", "Size"});
      if (!extra.bytes) {
        const JsonNode *args = row->Find({"Args", "args", "arguments"});
        if (args)
          extra.bytes = ToU64(ArgValue(FormatArgs(*args), "size"));
      }
      extra.device = NodeText(*row, {"Device", "device"});
      extra.queue =
          std::to_string(NodeInt(*row, {"Queue", "queue", "Queue_Id"}, 0));
      if (corr)
        copy_.Supplement(corr, extra);
      return;
    }
    if (section == ExtensionSection::Ops) {
      std::string op = Lower(NodeText(*row, {"Op", "op", "operation"}));
      const std::string kind = NodeText(*row, {"Kind", "kind", "direction"});
      const std::string lower_kind = Lower(kind);
      if (op.empty()) {
        if (lower_kind.find("barrier") != std::string::npos)
          op = "barrier";
        else if (lower_kind.find("fill") != std::string::npos ||
                 lower_kind.find("memset") != std::string::npos)
          op = "copy";
      }
      if (op == "dispatch" ||
          (op == "copy" && lower_kind.find("fill") == std::string::npos &&
           lower_kind.find("memset") == std::string::npos))
        return;
      if (op != "barrier" && op != "copy" && op != "other")
        return;
      double begin = NodeNumber(
          *row, {"Begin_us", "begin_us"},
          NodeInt(*row, {"Start_Timestamp", "start_timestamp", "begin_ns"}) /
              1000.0);
      double end = NodeNumber(
          *row, {"End_us", "end_us"},
          NodeInt(*row, {"End_Timestamp", "end_timestamp", "end_ns"}) / 1000.0);
      OpRow operation{
          op,
          kind,
          "",
          NodeText(*row, {"Device", "device", "Agent_Id", "agent_id"}),
          std::to_string(NodeInt(*row, {"Queue", "queue", "Queue_Id"})),
          NodeInt(*row, {"Bytes", "bytes", "size"}),
          NodeInt(*row, {"Correlation_Id", "correlation_id", "Corr", "corr"},
                  ToU64(key)),
          begin,
          end};
      ops_.Add(operation);
      phase_ops.push_back(std::move(operation));
      return;
    }
    if (section == ExtensionSection::Kernels) {
      KernelData data;
      data.name = NodeText(*row, {"Kernel_Name", "kernel_name", "name"});
      data.id = NodeInt(*row, {"Kernel_Id", "kernel_id", "id"}, ToU64(key));
      data.vgpr = NodeText(*row, {"VGPR", "vgpr", "VGPR_Count"});
      data.scratch =
          NodeText(*row, {"Scratch_Bytes", "scratch_bytes", "Scratch_Size"});
      data.lds = NodeText(*row, {"Static_LDS", "static_lds", "LDS_Block_Size"});
      data.occupancy =
          NodeText(*row, {"Theoretical_Occupancy_Pct",
                          "theoretical_occupancy_pct", "occupancy_pct"});
      if (!data.name.empty())
        kernels_by_name[data.name] = data;
      if (data.id)
        kernels_by_id[data.id] = data;
      return;
    }
    if (section == ExtensionSection::Vram) {
      double time = NodeNumber(
          *row, {"Time_us", "time_us"},
          NodeInt(*row, {"Timestamp", "timestamp", "timestamp_ns", "time_ns"}) /
              1000.0);
      const uint64_t total =
          NodeInt(*row, {"Total_Bytes", "total_bytes", "total"});
      const uint64_t free = NodeInt(*row, {"Free_Bytes", "free_bytes", "free"});
      const uint64_t used = NodeInt(*row, {"Used_Bytes", "used_bytes", "used"},
                                    total >= free ? total - free : 0);
      vram_.Add(time, used);
    }
  }

  std::unordered_map<std::string, KernelData> kernels_by_name;
  std::unordered_map<uint64_t, KernelData> kernels_by_id;
  std::vector<OpRow> phase_ops;

private:
  OpsAggregate &ops_;
  ApiJoin &api_;
  CopyJoin &copy_;
  VramAggregate &vram_;
};

class ExtensionHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ExtensionHandler> {
public:
  explicit ExtensionHandler(ExtensionConsumer &consumer)
      : consumer_(consumer) {}

  bool Null() {
    Scalar([&] { builder_.Null(); });
    return true;
  }
  bool Bool(bool value) {
    Scalar([&] { builder_.Bool(value); });
    return true;
  }
  bool Int(int value) { return Int64(value); }
  bool Uint(unsigned value) { return Uint64(value); }
  bool Int64(int64_t value) {
    Scalar([&] { builder_.Number(std::to_string(value)); });
    return true;
  }
  bool Uint64(uint64_t value) {
    Scalar([&] { builder_.Number(std::to_string(value)); });
    return true;
  }
  bool Double(double value) {
    Scalar([&] {
      std::ostringstream out;
      out << std::setprecision(17) << value;
      builder_.Number(out.str());
    });
    return true;
  }
  bool RawNumber(const char *value, rapidjson::SizeType length, bool) {
    Scalar([&] { builder_.Number(std::string(value, length)); });
    return true;
  }
  bool String(const char *value, rapidjson::SizeType length, bool) {
    Scalar([&] { builder_.String(value, length); });
    return true;
  }
  bool Key(const char *value, rapidjson::SizeType length, bool) {
    if (capturing_)
      builder_.Key(value, length);
    else if (active_ && object_container_ && depth_ == container_depth_)
      member_key_.assign(value, length);
    else if (!active_ && depth_ == 1) {
      pending_section_ = ClassifySection(std::string(value, length));
    }
    return true;
  }
  bool StartObject() { return Start(true); }
  bool StartArray() { return Start(false); }
  bool EndObject(rapidjson::SizeType) { return End(); }
  bool EndArray(rapidjson::SizeType) { return End(); }

private:
  template <typename F> void Scalar(F add) {
    if (capturing_) {
      add();
      return;
    }
    if (!active_ || depth_ != container_depth_)
      return;
    builder_.Reset();
    add();
    consumer_.Item(section_, object_container_ ? member_key_ : std::string{},
                   builder_.Take());
    member_key_.clear();
  }

  bool Start(bool object) {
    if (capturing_) {
      object ? builder_.StartObject() : builder_.StartArray();
      ++depth_;
      return true;
    }
    if (active_ && depth_ == container_depth_ &&
        (!object_container_ || !member_key_.empty())) {
      capturing_ = true;
      builder_.Reset();
      object ? builder_.StartObject() : builder_.StartArray();
      ++depth_;
      return true;
    }
    if (!active_ && depth_ == 1 && pending_section_ != ExtensionSection::None) {
      active_ = true;
      section_ = pending_section_;
      object_container_ = object;
      container_depth_ = depth_ + 1;
      pending_section_ = ExtensionSection::None;
    }
    ++depth_;
    return true;
  }

  bool End() {
    if (capturing_) {
      builder_.End();
      --depth_;
      if (depth_ == container_depth_) {
        capturing_ = false;
        consumer_.Item(section_,
                       object_container_ ? member_key_ : std::string{},
                       builder_.Take());
        member_key_.clear();
      }
      return true;
    }
    if (active_ && depth_ == container_depth_) {
      --depth_;
      active_ = false;
      section_ = ExtensionSection::None;
      member_key_.clear();
      return true;
    }
    --depth_;
    return true;
  }

  ExtensionConsumer &consumer_;
  NodeBuilder builder_;
  int depth_ = 0;
  int container_depth_ = 0;
  bool active_ = false;
  bool object_container_ = false;
  bool capturing_ = false;
  ExtensionSection section_ = ExtensionSection::None;
  ExtensionSection pending_section_ = ExtensionSection::None;
  std::string member_key_;
};

void ReadExtensions(const fs::path &path, ExtensionConsumer &consumer) {
  if (!fs::exists(path))
    return;
  FILE *file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"rb") || !file)
    throw std::runtime_error("cannot open " + path.string());
  struct CloseFile {
    void operator()(FILE *f) const {
      if (f)
        std::fclose(f);
    }
  };
  std::unique_ptr<FILE, CloseFile> owner(file);
  std::array<char, 65536> buffer{};
  rapidjson::FileReadStream stream(file, buffer.data(), buffer.size());
  rapidjson::Reader reader;
  ExtensionHandler handler(consumer);
  const rapidjson::ParseResult result = reader.Parse(stream, handler);
  if (!result)
    throw std::runtime_error(path.string() + ": JSON parse error at byte " +
                             std::to_string(result.Offset()) + ": " +
                             rapidjson::GetParseError_En(result.Code()));
}

std::unordered_map<std::string, KernelData>
ReadKernelData(const Capture &capture, const ExtensionConsumer *extensions,
               std::unordered_map<std::string, KernelData> standard_rows = {}) {
  std::unordered_map<std::string, KernelData> result = std::move(standard_rows);
  const fs::path path = capture.Domain("kernels");
  if (!fs::exists(path))
    return result;
  if (!capture.IsLegacy()) {
    if (!extensions)
      return result;
    for (auto &[name, data] : result) {
      const auto by_name = extensions->kernels_by_name.find(name);
      const auto by_id = extensions->kernels_by_id.find(data.id);
      const KernelData *extra =
          by_name != extensions->kernels_by_name.end() ? &by_name->second
          : by_id != extensions->kernels_by_id.end()   ? &by_id->second
                                                       : nullptr;
      if (!extra)
        continue;
      if (!extra->vgpr.empty())
        data.vgpr = extra->vgpr;
      if (!extra->scratch.empty())
        data.scratch = extra->scratch;
      if (!extra->lds.empty())
        data.lds = extra->lds;
      data.occupancy = extra->occupancy;
    }
    return result;
  }
  CsvReader reader(path);
  CsvRow row;
  while (reader.Next(row)) {
    const std::string name = Get(row, "Kernel_Name");
    if (name.empty() || result.count(name))
      continue;
    result[name] = {name,
                    ToU64(Get(row, "Kernel_Id")),
                    Get(row, "VGPR"),
                    Get(row, "Scratch_Bytes"),
                    Get(row, "Static_LDS"),
                    Get(row, "Theoretical_Occupancy_Pct")};
  }
  return result;
}

Capture ResolveAnalyzeCapture(const fs::path &source) {
  const std::string raw = source.string();
  for (const std::string suffix :
       {".api.csv", ".kernels.csv", ".vram.csv", ".summary.txt"}) {
    if (!EndsWithInsensitive(raw, suffix))
      continue;
    const fs::path ops(raw.substr(0, raw.size() - suffix.size()));
    if (fs::exists(ops) && EndsWithInsensitive(ops.string(), "_ops.csv"))
      return {fs::path(ops.string().substr(0, ops.string().size() - 8)), source,
              ops};
  }
  if (!fs::exists(source) && fs::exists(fs::path(raw + ".csv")) &&
      EndsWithInsensitive(raw + ".csv", "_ops.csv")) {
    const fs::path ops(raw + ".csv");
    return {fs::path(ops.string().substr(0, ops.string().size() - 8)), source,
            ops};
  }
  return ResolveCapture(source);
}

template <typename Map>
std::vector<std::pair<typename Map::key_type, typename Map::mapped_type>>
SortedValues(const Map &values) {
  std::vector<std::pair<typename Map::key_type, typename Map::mapped_type>>
      result;
  result.reserve(values.size());
  for (const auto &[key, value] : values)
    result.emplace_back(key, value);
  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  });
  return result;
}

std::string Label(const std::unordered_map<std::string, std::string> &labels,
                  const std::string &name) {
  const auto it = labels.find(name);
  return it == labels.end() ? name : it->second;
}

void Section(std::ostringstream &out, const std::string &title) {
  out << '\n' << title << '\n' << std::string(title.size(), '-') << '\n';
}

std::string
WriteReport(const OpsAggregate &ops, const ApiAggregate *api,
            const std::unordered_map<std::string, KernelData> &kernels,
            const VramAggregate &vram,
            const std::unordered_map<std::string, std::string> &labels,
            const std::string &device_name, const RuntimeInfo &runtime,
            int top) {
  std::ostringstream out;
  const double span = ops.Span();
  Section(out, "overview");
  out << "  ktrace version: " KTRACE_VERSION_STRING "\n";
  if (!device_name.empty())
    out << "  device        : " << device_name << '\n';
  if (!runtime.path.empty())
    out << "  amdhip64 path : " << runtime.path << '\n';
  if (!runtime.version.empty())
    out << "  amdhip64 ver  : " << runtime.version << '\n';
  uint64_t commands = 0;
  for (const auto &[_, count] : ops.counts)
    commands += count;
  const auto count = [&](const char *name) {
    const auto it = ops.counts.find(name);
    return it == ops.counts.end() ? 0ull : it->second;
  };
  out << "  commands      : " << Commas(commands) << "  (dispatch "
      << Commas(count("dispatch")) << ", copy " << Commas(count("copy"))
      << ", barrier " << Commas(count("barrier")) << ")\n";
  out << "  span          : " << Fixed(span / 1e6, 2) << " s\n";
  out << "  GPU busy      : " << Fixed(ops.busy / 1e6, 2) << " s ("
      << Fixed(span ? 100 * ops.busy / span : 0, 1) << "% of span)\n";
  out << "  queues        : " << ops.queues.size() << '\n';
  std::vector<std::pair<std::string, CountTime>> queues(ops.queues.begin(),
                                                        ops.queues.end());
  std::sort(queues.begin(), queues.end(), [](const auto &a, const auto &b) {
    return a.second.us > b.second.us;
  });
  double queue_busy = 0;
  for (const auto &[name, value] : queues) {
    queue_busy += value.us;
    out << "    " << Left(name, 8) << ' ' << Right(Commas(value.count), 9)
        << " cmds  busy " << Right(Fixed(value.us / 1e6, 2), 7) << " s ("
        << Right(Fixed(span ? 100 * value.us / span : 0, 1), 5)
        << "% of span)\n";
  }
  const double overlap = queue_busy ? ops.busy / queue_busy : 0;
  if (std::abs(overlap - 1.0) < 0.01 && queues.size() > 1)
    out << "    queues never overlapped: the work is serialised across them\n";

  if (!ops.phases.empty()) {
    Section(out, "phases (inferred from kernel names)");
    const double first = ops.phases.front().begin_us;
    std::map<std::string, CountTime> runs;
    for (const auto &phase : ops.phases) {
      ++runs[phase.tag].count;
      runs[phase.tag].us += phase.end_us - phase.begin_us;
      out << "  " << Left(phase.tag, 9)
          << Right(Fixed((phase.begin_us - first) / 1e6, 2), 8) << " .. "
          << Right(Fixed((phase.end_us - first) / 1e6, 2), 6) << " s   ("
          << Fixed((phase.end_us - phase.begin_us) / 1e6, 2) << " s)\n";
    }
    out << '\n'
        << "  " << Left("phase", 9) << Right("runs", 6) << Right("wall_s", 9)
        << Right("gpu_s", 9) << Right("busy", 7);
    if (api)
      out << Right("host_wait_s", 13);
    out << '\n';
    std::vector<std::pair<std::string, CountTime>> ordered(runs.begin(),
                                                           runs.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
      return a.second.us > b.second.us;
    });
    for (const auto &[tag, value] : ordered) {
      const auto gpu_it = ops.phase_busy.find(tag);
      const double gpu =
          gpu_it == ops.phase_busy.end() ? 0 : gpu_it->second / 1e6;
      out << "  " << Left(tag, 9) << Right(std::to_string(value.count), 6)
          << Right(Fixed(value.us / 1e6, 2), 9) << Right(Fixed(gpu, 2), 9)
          << Right(Fixed(value.us ? 100 * gpu / (value.us / 1e6) : 0, 0) + "%",
                   7);
      if (api) {
        const auto wait = api->phase_wait.find(tag);
        out << Right(
            Fixed((wait == api->phase_wait.end() ? 0 : wait->second) / 1e6, 2),
            13);
      }
      out << '\n';
    }
    std::vector<std::pair<std::string, double>> phase_order(
        ops.phase_busy.begin(), ops.phase_busy.end());
    std::sort(phase_order.begin(), phase_order.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    for (const auto &[tag, _] : phase_order) {
      out << "  " << tag << " is mostly: ";
      const auto found = ops.phase_kernel.find(tag);
      auto hot = SortedValues(found->second);
      for (size_t i = 0; i < hot.size() && i < 3; ++i) {
        if (i)
          out << ", ";
        out << Label(labels, hot[i].first).substr(0, 38) << " ("
            << Fixed(hot[i].second / 1e6, 2) << " s)";
      }
      out << '\n';
    }
  }

  Section(out, "top " + std::to_string(top) + " kernels by GPU time");
  out << "  " << Left("kernel", 44) << Right("total_s", 9)
      << Right("launches", 10) << Right("avg_us", 10);
  if (!kernels.empty())
    out << Right("occ", 8) << Right("VGPR", 6) << Right("scratch", 9)
        << Right("LDS", 8);
  out << '\n';
  auto hot_kernels = SortedValues(ops.kernel_time);
  for (size_t i = 0;
       i < hot_kernels.size() && i < static_cast<size_t>(std::max(top, 0));
       ++i) {
    const auto &[name, us] = hot_kernels[i];
    const uint64_t launches = ops.kernel_count.at(name);
    out << "  " << Left(Label(labels, name).substr(0, 44), 44)
        << Right(Fixed(us / 1e6, 2), 9) << Right(Commas(launches), 10)
        << Right(Fixed(launches ? us / launches : 0, 1), 10);
    if (!kernels.empty()) {
      const auto found = kernels.find(name);
      if (found == kernels.end())
        out << Right("-", 8) << Right("-", 6) << Right("-", 9) << Right("-", 8);
      else
        out << Right(found->second.occupancy.empty()
                         ? "-"
                         : found->second.occupancy + "%",
                     8)
            << Right(found->second.vgpr.empty() ? "0" : found->second.vgpr, 6)
            << Right(found->second.scratch.empty() ? "0"
                                                   : found->second.scratch,
                     9)
            << Right(found->second.lds.empty() ? "0" : found->second.lds, 8);
    }
    out << '\n';
  }

  if (!kernels.empty()) {
    Section(out, "kernels whose static footprint limits them");
    std::vector<std::pair<std::string, double>> spills, low;
    for (const auto &[name, us] : ops.kernel_time) {
      const auto found = kernels.find(name);
      if (found == kernels.end())
        continue;
      if (ToU64(found->second.scratch) > 0)
        spills.emplace_back(name, us);
      const double occupancy = ToDouble(found->second.occupancy);
      if (!found->second.occupancy.empty() && occupancy > 0 &&
          occupancy < kLowOccupancyPct)
        low.emplace_back(name, us);
    }
    const auto descending = [](const auto &a, const auto &b) {
      return a.second > b.second;
    };
    std::sort(spills.begin(), spills.end(), descending);
    std::sort(low.begin(), low.end(), descending);
    if (!spills.empty()) {
      double total = 0;
      for (const auto &item : spills)
        total += item.second;
      out << "  register spill  : " << spills.size() << " kernels, "
          << Fixed(ops.busy ? 100 * total / ops.busy : 0, 1)
          << "% of GPU time\n";
      for (size_t i = 0; i < spills.size() && i < 6; ++i) {
        const auto &data = kernels.at(spills[i].first);
        out << "    " << Left(Label(labels, spills[i].first).substr(0, 44), 44)
            << Right(Fixed(spills[i].second / 1e6, 2), 8) << " s  scratch "
            << Right(data.scratch, 6) << " B  VGPR " << Right(data.vgpr, 4)
            << '\n';
      }
    }
    if (!low.empty()) {
      double total = 0;
      for (const auto &item : low)
        total += item.second;
      out << "  occupancy <50%  : " << low.size() << " kernels, "
          << Fixed(ops.busy ? 100 * total / ops.busy : 0, 1)
          << "% of GPU time\n";
      for (size_t i = 0; i < low.size() && i < 6; ++i) {
        const auto &data = kernels.at(low[i].first);
        const std::string why = ToU64(data.vgpr) >= 128   ? "registers"
                                : ToU64(data.lds) > 16384 ? "LDS"
                                                          : "block size";
        out << "    " << Left(Label(labels, low[i].first).substr(0, 44), 44)
            << Right(Fixed(low[i].second / 1e6, 2), 8) << " s  occ "
            << Right(data.occupancy, 5) << "%  limited by " << why << '\n';
      }
    }
    if (spills.empty() && low.empty())
      out << "  nothing flagged\n";
  }

  if (api) {
    Section(out, "top " + std::to_string(top) + " HIP calls by host time");
    // Widths hold the longest real cases: hipOccupancyMaxActiveBlocksPer-
    // Multiprocessor is 43 characters, and a first hipMalloc of over 100 ms
    // prints ten digits of microseconds, which left no separating space.
    out << "  " << Left("api", 46) << Right("count", 12) << Right("total_s", 10)
        << Right("avg_us", 13) << '\n';
    std::vector<std::pair<std::string, CountTime>> calls(api->per_api.begin(),
                                                         api->per_api.end());
    std::sort(calls.begin(), calls.end(), [](const auto &a, const auto &b) {
      return a.second.us > b.second.us;
    });
    for (size_t i = 0;
         i < calls.size() && i < static_cast<size_t>(std::max(top, 0)); ++i)
      out << "  " << Left(calls[i].first, 46)
          << Right(Commas(calls[i].second.count), 12)
          << Right(Fixed(calls[i].second.us / 1e6, 2), 10)
          << Right(Fixed(calls[i].second.count
                             ? calls[i].second.us / calls[i].second.count
                             : 0,
                         3),
                   13)
          << '\n';

    if (!api->stream_wait.empty()) {
      Section(out, "host wait, by the queue it waited on");
      out << "  " << Left("stream", 20) << Left("hw queue", 10)
          << Right("sync calls", 12) << Right("wait_s", 10) << '\n';
      std::vector<std::pair<std::string, CountTime>> waits(
          api->stream_wait.begin(), api->stream_wait.end());
      std::sort(waits.begin(), waits.end(), [](const auto &a, const auto &b) {
        return a.second.us > b.second.us;
      });
      for (const auto &[stream, wait] : waits) {
        std::string queue = "?";
        const auto mapped = api->stream_queue.find(stream);
        if (mapped != api->stream_queue.end() && !mapped->second.empty()) {
          queue = std::max_element(mapped->second.begin(), mapped->second.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.second < b.second;
                                   })
                      ->first;
        }
        out << "  " << Left(stream, 20) << Left(queue, 10)
            << Right(Commas(wait.count), 12)
            << Right(Fixed(wait.us / 1e6, 2), 10) << '\n';
      }
    }
  }

  Section(out, "memory");
  if (!ops.transfers.empty()) {
    // MEMORY_COPY_DEVICE_TO_DEVICE is 28 characters and overflowed 24,
    // which shifted the count column on that row only.
    out << "  " << Left("transfer", 32) << Right("count", 10)
        << Right("total", 12) << Right("GB/s", 9) << '\n';
    std::vector<std::pair<std::string, Transfer>> transfers(
        ops.transfers.begin(), ops.transfers.end());
    std::sort(transfers.begin(), transfers.end(),
              [](const auto &a, const auto &b) {
                return a.second.bytes > b.second.bytes;
              });
    for (const auto &[kind, value] : transfers)
      out << "  " << Left(kind, 32) << Right(Commas(value.count), 10)
          << Right(HumanBytes(static_cast<double>(value.bytes)), 12)
          << Right(Fixed(value.us ? value.bytes / (value.us * 1000.0) : 0, 1),
                   9)
          << '\n';
  }
  if (api) {
    out << "  requested by the application (from hipMalloc/hipFree):\n"
        << "    peak held   : "
        << HumanBytes(static_cast<double>(api->alloc_peak)) << '\n'
        << "    cumulative  : "
        << HumanBytes(static_cast<double>(api->alloc_total)) << '\n'
        << "    still held  : "
        << HumanBytes(static_cast<double>(api->alloc_current)) << " in "
        << api->allocations.size() << " blocks";
    if (api->unmatched_free)
      out << ", " << api->unmatched_free << " unmatched frees";
    out << '\n';
  }
  if (vram.count) {
    out << "  reported by the driver (" << vram.count << " samples over "
        << Fixed((vram.last_time - vram.first_time) / 1e6, 1) << " s):\n"
        << "    peak used   : " << HumanBytes(static_cast<double>(vram.maximum))
        << '\n'
        << "    growth      : "
        << HumanBytes(static_cast<double>(static_cast<int64_t>(vram.last) -
                                          static_cast<int64_t>(vram.first)))
        << '\n';
    if (api && api->alloc_peak) {
      const int64_t gap = static_cast<int64_t>(vram.maximum) -
                          static_cast<int64_t>(api->alloc_peak);
      out << "    over what the application asked for: "
          << HumanBytes(static_cast<double>(gap))
          << " (runtime buffers, pools, fragmentation)\n";
    }
  }
  out << '\n';
  return out.str();
}

std::string WriteRoctxReport(const RoctxAggregate &roctx, int top) {
  if (!roctx.events)
    return {};
  std::ostringstream out;
  Section(out, "ROCTx annotations");
  uint64_t completed = 0;
  for (const auto &[_, range] : roctx.ranges)
    completed += range.count;
  out << "  events        : " << Commas(roctx.events) << '\n'
      << "  marks         : " << Commas(roctx.marks) << '\n'
      << "  ranges        : " << Commas(completed) << '\n'
      << "  unmatched     : " << Commas(roctx.unmatched) << '\n';

  if (!roctx.ranges.empty()) {
    out << '\n'
        << "  " << Left("range", 38) << Right("count", 9)
        << Right("total_ms", 12) << Right("avg_ms", 11) << Right("max_ms", 11)
        << Right("cross", 8) << '\n';
    std::vector<std::pair<std::string, RoctxRangeStats>> ranges(
        roctx.ranges.begin(), roctx.ranges.end());
    std::sort(ranges.begin(), ranges.end(),
              [](const auto &left, const auto &right) {
                return left.second.total_us > right.second.total_us;
              });
    for (size_t i = 0;
         i < ranges.size() && i < static_cast<size_t>(std::max(top, 0)); ++i) {
      const auto &[name, range] = ranges[i];
      out << "  " << Left(name, 38) << Right(Commas(range.count), 9)
          << Right(Fixed(range.total_us / 1000.0, 3), 12)
          << Right(
                 Fixed(range.count ? range.total_us / range.count / 1000.0 : 0,
                       3),
                 11)
          << Right(Fixed(range.max_us / 1000.0, 3), 11)
          << Right(Commas(range.cross_thread), 8) << '\n';
    }
  }
  out << '\n';
  return out.str();
}
} // namespace

int AnalyzeCommand(const std::vector<std::string> &args) {
  if (args.empty() || args[0] == "-h" || args[0] == "--help") {
    std::cout
        << "usage: ktrace_postprocess.exe analyze <capture> [--top N] [-o "
           "FILE] [--quiet]\n"
           "  --top N   rows in the kernel and HIP call tables (default 20)\n"
           "  capture: rocprofv3 prefix/domain/results file, or legacy "
           "*_ops.csv\n";
    return args.empty() ? 2 : 0;
  }

  std::optional<fs::path> source;
  std::optional<std::string> output;
  int top = 20;
  bool quiet = false;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--top") {
      if (++i == args.size())
        throw std::runtime_error("--top requires a value");
      top = std::stoi(args[i]);
    } else if (args[i] == "-o" || args[i] == "--out") {
      if (++i == args.size())
        throw std::runtime_error(args[i - 1] + " requires a value");
      output = args[i];
    } else if (args[i] == "--quiet")
      quiet = true;
    else if (!args[i].empty() && args[i][0] == '-')
      throw std::runtime_error("unknown analyze option: " + args[i]);
    else if (source)
      throw std::runtime_error("analyze accepts exactly one capture");
    else
      source = fs::u8path(args[i]);
  }
  if (!source)
    throw std::runtime_error("analyze requires a capture");

  const Capture capture = ResolveAnalyzeCapture(*source);
  OpsAggregate ops;
  ApiAggregate api;
  VramAggregate vram;
  bool has_api = false;
  std::unordered_map<std::string, KernelData> kernels;

  if (capture.IsLegacy()) {
    ReadLegacyOps(capture.Domain("ops"), ops);
    ops.FinishPhases();
    ReadLegacyPhaseOps(capture.Domain("ops"), ops);
    has_api = fs::exists(capture.Domain("api"));
    if (has_api)
      ReadLegacyApi(capture, ops, api);
    ReadLegacyVram(capture.Domain("vram"), vram);
    kernels = ReadKernelData(capture, nullptr);
  } else {
    std::unordered_map<std::string, KernelData> standard_kernels;
    ReadKernelOps(capture.Domain("kernels"), ops, standard_kernels);
    ops.FinishPhases();
    CopyJoin copy(capture, ops);
    ApiJoin api_join(capture, ops, api);
    has_api = api_join.IsOpen();
    ExtensionConsumer extensions(ops, api_join, copy, vram);
    ReadExtensions(capture.Domain("extensions"), extensions);
    copy.Finish();
    api_join.Finish();
    ReadKernelPhaseOps(capture.Domain("kernels"), ops);
    ReadCopyPhaseOps(capture.Domain("copy"), ops);
    for (const auto &operation : extensions.phase_ops)
      ops.AddPhase(operation);
    if (has_api)
      ReadAllocations(capture.Domain("alloc"), api);
    kernels = ReadKernelData(capture, &extensions, std::move(standard_kernels));
  }

  std::vector<std::string> raw_names;
  raw_names.reserve(ops.kernel_time.size());
  for (const auto &[name, _] : ops.kernel_time)
    raw_names.push_back(name);
  const auto labels = ResolveKernelNames(raw_names);
  const RoctxAggregate roctx = ReadRoctx(capture);
  const std::string device_name = ReadDeviceName(capture);
  const RuntimeInfo runtime = ReadRuntimeInfo(capture);
  const std::string report =
      WriteReport(ops, has_api ? &api : nullptr, kernels, vram, labels,
                  device_name, runtime, top) +
      WriteRoctxReport(roctx, top);
  if (!quiet)
    std::cout << report;

  std::optional<fs::path> report_path;
  if (!output || *output != "-")
    report_path =
        output ? fs::u8path(*output) : capture.AutoOutput("_ktrace_report.txt");
  if (report_path) {
    std::ofstream file(*report_path, std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot create " + report_path->string());
    file << report;
    file.close();
    if (!quiet)
      std::cout << "-> " << report_path->string() << '\n';
  }
  return 0;
}
} // namespace ktrace
