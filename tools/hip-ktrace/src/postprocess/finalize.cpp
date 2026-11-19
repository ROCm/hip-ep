/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ktrace {
namespace {
constexpr uint32_t kMemoryCopyKind = 10;
constexpr uint32_t kKernelDispatchKind = 11;
constexpr uint32_t kMemoryAllocationKind = 16;
constexpr uint32_t kHipRuntimeKind = 21;
constexpr uint32_t kHipCompilerKind = 22;

const std::vector<std::string> kAgentHeader = {"Node_Id",
                                               "Logical_Node_Id",
                                               "Agent_Type",
                                               "Cpu_Cores_Count",
                                               "Simd_Count",
                                               "Cpu_Core_Id_Base",
                                               "Simd_Id_Base",
                                               "Max_Waves_Per_Simd",
                                               "Lds_Size_In_Kb",
                                               "Gds_Size_In_Kb",
                                               "Num_Gws",
                                               "Wave_Front_Size",
                                               "Num_Xcc",
                                               "Cu_Count",
                                               "Array_Count",
                                               "Num_Shader_Banks",
                                               "Simd_Arrays_Per_Engine",
                                               "Cu_Per_Simd_Array",
                                               "Simd_Per_Cu",
                                               "Max_Slots_Scratch_Cu",
                                               "Gfx_Target_Version",
                                               "Vendor_Id",
                                               "Device_Id",
                                               "Location_Id",
                                               "Domain",
                                               "Drm_Render_Minor",
                                               "Num_Sdma_Engines",
                                               "Num_Sdma_Xgmi_Engines",
                                               "Num_Sdma_Queues_Per_Engine",
                                               "Num_Cp_Queues",
                                               "Max_Engine_Clk_Ccompute",
                                               "Max_Engine_Clk_Fcompute",
                                               "Sdma_Fw_Version",
                                               "Fw_Version",
                                               "Capability",
                                               "Cu_Per_Engine",
                                               "Max_Waves_Per_Cu",
                                               "Family_Id",
                                               "Workgroup_Max_Size",
                                               "Grid_Max_Size",
                                               "Local_Mem_Size",
                                               "Hive_Id",
                                               "Gpu_Id",
                                               "Workgroup_Max_Dim_X",
                                               "Workgroup_Max_Dim_Y",
                                               "Workgroup_Max_Dim_Z",
                                               "Grid_Max_Dim_X",
                                               "Grid_Max_Dim_Y",
                                               "Grid_Max_Dim_Z",
                                               "Name",
                                               "Vendor_Name",
                                               "Product_Name",
                                               "Model_Name"};
const std::vector<std::string> kHipHeader = {
    "Domain",         "Function",        "Process_Id",   "Thread_Id",
    "Correlation_Id", "Start_Timestamp", "End_Timestamp"};
const std::vector<std::string> kKernelHeader = {"Kind",
                                                "Agent_Id",
                                                "Queue_Id",
                                                "Stream_Id",
                                                "Thread_Id",
                                                "Dispatch_Id",
                                                "Kernel_Id",
                                                "Kernel_Name",
                                                "Correlation_Id",
                                                "Start_Timestamp",
                                                "End_Timestamp",
                                                "LDS_Block_Size",
                                                "Scratch_Size",
                                                "VGPR_Count",
                                                "Accum_VGPR_Count",
                                                "SGPR_Count",
                                                "Workgroup_Size_X",
                                                "Workgroup_Size_Y",
                                                "Workgroup_Size_Z",
                                                "Grid_Size_X",
                                                "Grid_Size_Y",
                                                "Grid_Size_Z"};
const std::vector<std::string> kCopyHeader = {"Kind",
                                              "Direction",
                                              "Stream_Id",
                                              "Source_Agent_Id",
                                              "Destination_Agent_Id",
                                              "Correlation_Id",
                                              "Start_Timestamp",
                                              "End_Timestamp"};
const std::vector<std::string> kAllocHeader = {
    "Kind",    "Operation",      "Agent_Id",        "Allocation_Size",
    "Address", "Correlation_Id", "Start_Timestamp", "End_Timestamp"};
const std::vector<std::string> kRoctxHeader = {
    "Kind",     "Operation", "Process_Id", "Thread_Id",
    "Range_Id", "Timestamp", "Message"};

struct Summary {
  uint64_t pid = 0;
  std::string device_name = "AMD GPU";
  int64_t cu_count = 0;
  int64_t wave_front_size = 0;
  int64_t threads_per_cu = 0;
  int64_t local_mem_size = 0;
  int64_t clock_khz = 0;
  int64_t memory_clock_khz = 0;
  int64_t memory_bus_width = 0;
  std::string text;
};

struct KernelProperty {
  CsvRow row;
  int vgpr = 0;
  uint64_t scratch = 0;
  uint64_t static_lds = 0;
};

#pragma pack(push, 1)
struct CorrelationMeta {
  uint32_t api_id = 0;
  uint32_t tid = 0;
  uint64_t stream = 0;
  uint32_t gx = 0, gy = 0, gz = 0;
  uint32_t bx = 0, by = 0, bz = 0;
  uint64_t dynamic_lds = 0;
};
#pragma pack(pop)
static_assert(sizeof(CorrelationMeta) == 48,
              "correlation metadata layout changed");

struct CopyDirection {
  const char *name;
  uint32_t operation;
  uint64_t source_agent;
  uint64_t destination_agent;
};

const std::unordered_map<std::string, CopyDirection> kCopyDirections = {
    {"CopyHostToDevice", {"MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2}},
    {"CopyHostToDevice2D", {"MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2}},
    {"WriteImage", {"MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2}},
    {"CopyDeviceToHost", {"MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1}},
    {"CopyDeviceToHost2D", {"MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1}},
    {"ReadImage", {"MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1}},
    {"CopyDeviceToDevice", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
    {"CopyDeviceToDevice2D", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
    {"CopyImage", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
    {"CopyImageToBuffer", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
    {"CopyBufferToImage", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
    {"SvmMemcpy", {"MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2}},
};

std::string Get(const CsvRow &row, const std::string &key) {
  const auto it = row.find(key);
  return it == row.end() ? std::string{} : it->second;
}

bool Complete(const CsvRow &row, std::initializer_list<const char *> required) {
  for (const char *key : required)
    if (Get(row, key).empty())
      return false;
  return true;
}

uint64_t Nanoseconds(const std::string &value) {
  const double result = std::round(ToDouble(value) * 1000.0);
  return result > 0 ? static_cast<uint64_t>(result) : 0;
}

std::string U64(uint64_t value) { return std::to_string(value); }
std::string I64(int64_t value) { return std::to_string(value); }

std::string HexAddress(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << std::setw(16)
      << std::setfill('0') << value;
  return out.str();
}

std::string Trim(std::string value) {
  while (!value.empty() && (value.back() == '\r' || value.back() == '\n' ||
                            value.back() == ' ' || value.back() == '\t'))
    value.pop_back();
  size_t first = 0;
  while (first < value.size() && (value[first] == ' ' || value[first] == '\t'))
    ++first;
  return value.substr(first);
}

std::string AfterColon(const std::string &line) {
  const size_t pos = line.find(':');
  return pos == std::string::npos ? std::string{} : Trim(line.substr(pos + 1));
}

Summary ReadSummary(const fs::path &path) {
  Summary result;
  result.pid = GetCurrentProcessId();
  if (!fs::exists(path))
    return result;

  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot read summary: " + path.string());
  result.text.assign(std::istreambuf_iterator<char>(file), {});
  result.text = std::regex_replace(result.text, std::regex("\r\n"), "\n");
  result.text = std::regex_replace(result.text, std::regex("\r"), "\n");

  std::istringstream lines(result.text);
  std::string raw;
  const std::regex clock_pattern(
      R"(^clock / mem\s*:\s*(-?\d+) kHz / (-?\d+) kHz, bus (-?\d+) bit)");
  while (std::getline(lines, raw)) {
    const std::string line = Trim(raw);
    if (line.rfind("pid", 0) == 0)
      result.pid = ToU64(AfterColon(line), result.pid);
    else if (line.rfind("device", 0) == 0) {
      const size_t colon = line.find(':');
      const size_t first = line.find_first_not_of(" \t", 6);
      if (colon != std::string::npos && first == colon) {
        const std::string name = AfterColon(line);
        if (!name.empty())
          result.device_name = name;
      }
    } else if (line.rfind("CUs", 0) == 0)
      result.cu_count = ToI64(AfterColon(line));
    else if (line.rfind("warp size", 0) == 0)
      result.wave_front_size = ToI64(AfterColon(line));
    else if (line.rfind("threads/CU", 0) == 0)
      result.threads_per_cu = ToI64(AfterColon(line));
    else if (line.rfind("LDS/CU", 0) == 0)
      result.local_mem_size = ToI64(AfterColon(line));
    else {
      std::smatch match;
      if (std::regex_search(line, match, clock_pattern)) {
        result.clock_khz = ToI64(match[1].str());
        result.memory_clock_khz = ToI64(match[2].str());
        result.memory_bus_width = ToI64(match[3].str());
      }
    }
  }
  return result;
}

std::map<std::string, KernelProperty>
ReadKernelProperties(const fs::path &path, std::vector<CsvRow> &rows) {
  std::map<std::string, KernelProperty> result;
  if (!fs::exists(path))
    return result;
  CsvReader reader(path);
  if (!reader.IsOpen())
    throw std::runtime_error("cannot read kernel properties: " + path.string());
  CsvRow row;
  while (reader.Next(row)) {
    const std::string name = Get(row, "Kernel_Name");
    if (name.empty())
      continue;
    KernelProperty prop;
    prop.row = row;
    prop.vgpr = static_cast<int>(ToI64(Get(row, "VGPR")));
    prop.scratch = ToU64(Get(row, "Scratch_Bytes"));
    prop.static_lds = ToU64(Get(row, "Static_LDS"));
    rows.push_back(row);
    result[name] = prop;
  }
  return result;
}

struct ParsedArguments {
  std::vector<std::pair<std::string, std::string>> ordered;
  std::unordered_map<std::string, std::string> values;

  size_t count(const std::string &key) const { return values.count(key); }
  const std::string &at(const std::string &key) const { return values.at(key); }
  bool empty() const { return ordered.empty(); }
  auto begin() const { return ordered.begin(); }
  auto end() const { return ordered.end(); }
};

ParsedArguments ParseArguments(const std::string &text) {
  ParsedArguments result;
  std::istringstream input(text);
  std::string item;
  while (input >> item) {
    const size_t pos = item.find('=');
    if (pos != std::string::npos) {
      std::string key = item.substr(0, pos);
      std::string value = item.substr(pos + 1);
      result.ordered.emplace_back(key, value);
      result.values[key] = value;
    }
  }
  return result;
}

std::array<uint32_t, 3> ParseDim(const std::string &text) {
  std::array<uint32_t, 3> result{};
  size_t begin = 0;
  for (size_t i = 0; i < 3; ++i) {
    const size_t end = text.find('x', begin);
    if ((i < 2 && end == std::string::npos) ||
        (i == 2 && end != std::string::npos))
      return {};
    const std::string part = text.substr(begin, end - begin);
    if (part.empty())
      return {};
    char *parsed_end = nullptr;
    const unsigned long value = std::strtoul(part.c_str(), &parsed_end, 0);
    if (parsed_end == part.c_str() || *parsed_end != '\0')
      return {};
    result[i] = static_cast<uint32_t>(value);
    begin = end == std::string::npos ? text.size() : end + 1;
  }
  return result;
}

template <typename Fn> std::string Json(Fn &&fn) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  fn(writer);
  if (!writer.IsComplete())
    throw std::runtime_error("internal JSON serialization failure");
  return std::string(buffer.GetString(), buffer.GetSize());
}

template <typename Writer>
void String(Writer &writer, const std::string &value) {
  writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

class JsonFragment {
public:
  explicit JsonFragment(fs::path path, bool object = false)
      : path_(std::move(path)), object_(object), buffer_(1024 * 1024) {
    file_.rdbuf()->pubsetbuf(buffer_.data(),
                             static_cast<std::streamsize>(buffer_.size()));
    file_.open(path_, std::ios::binary);
    if (!file_)
      throw std::runtime_error("cannot create JSON fragment: " +
                               path_.string());
  }

  ~JsonFragment() { Close(); }
  JsonFragment(const JsonFragment &) = delete;
  JsonFragment &operator=(const JsonFragment &) = delete;

  void Add(const std::string &json) {
    if (count_++)
      file_ << ',';
    file_ << json;
    if (!file_)
      throw std::runtime_error("failed writing JSON fragment: " +
                               path_.string());
  }

  void AddEntry(uint64_t key, const std::string &value) {
    if (count_++)
      file_ << ',';
    file_ << JsonEscape(std::to_string(key)) << ':' << JsonEscape(value);
    if (!file_)
      throw std::runtime_error("failed writing JSON fragment: " +
                               path_.string());
  }

  void Close() {
    if (file_.is_open())
      file_.close();
  }

  void CopyTo(std::ostream &output) {
    Close();
    output << (object_ ? '{' : '[');
    std::ifstream input(path_, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot reopen JSON fragment: " +
                               path_.string());
    std::vector<char> buffer(1024 * 1024);
    while (input) {
      input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      output.write(buffer.data(), input.gcount());
    }
    output << (object_ ? '}' : ']');
    if (!output)
      throw std::runtime_error("failed assembling JSON output");
  }

  const fs::path &Path() const { return path_; }

private:
  fs::path path_;
  bool object_ = false;
  uint64_t count_ = 0;
  std::vector<char> buffer_;
  std::ofstream file_;
};

uint32_t CpuCount() {
  const uint32_t count = std::thread::hardware_concurrency();
  return count ? count : 1;
}

std::string ProcessorName() {
  char value[256]{};
  const DWORD n =
      GetEnvironmentVariableA("PROCESSOR_IDENTIFIER", value, sizeof(value));
  return n > 0 && n < sizeof(value) ? std::string(value, n) : "CPU";
}

std::string MachineName() {
  char value[64]{};
  const DWORD n =
      GetEnvironmentVariableA("PROCESSOR_ARCHITECTURE", value, sizeof(value));
  return n > 0 && n < sizeof(value) ? std::string(value, n) : "AMD64";
}

std::pair<std::string, std::string> WindowsVersion() {
  using RtlGetVersionFn = LONG(WINAPI *)(OSVERSIONINFOW *);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  const auto get_version = reinterpret_cast<RtlGetVersionFn>(
      ntdll ? GetProcAddress(ntdll, "RtlGetVersion") : nullptr);
  OSVERSIONINFOEXW info{};
  info.dwOSVersionInfoSize = sizeof(info);
  if (!get_version ||
      get_version(reinterpret_cast<OSVERSIONINFOW *>(&info)) != 0)
    return {"", ""};
  std::string release;
  if (info.wProductType != VER_NT_WORKSTATION) {
    if (info.dwBuildNumber >= 26100)
      release = "2025Server";
    else if (info.dwBuildNumber >= 20348)
      release = "2022Server";
    else if (info.dwBuildNumber >= 17763)
      release = "2019Server";
    else if (info.dwBuildNumber >= 14393)
      release = "2016Server";
    else
      release = std::to_string(info.dwMajorVersion) + "Server";
  } else {
    release = info.dwBuildNumber >= 22000 ? "11"
                                          : std::to_string(info.dwMajorVersion);
  }
  const std::string version = std::to_string(info.dwMajorVersion) + "." +
                              std::to_string(info.dwMinorVersion) + "." +
                              std::to_string(info.dwBuildNumber);
  return {release, version};
}

uint64_t Positive(int64_t value, uint64_t fallback = 0) {
  return value > 0 ? static_cast<uint64_t>(value) : fallback;
}

template <typename Writer> void WriteHandle(Writer &writer, uint64_t value) {
  writer.StartObject();
  writer.Key("handle");
  writer.Uint64(value);
  writer.EndObject();
}

template <typename Writer>
void WriteDim(Writer &writer, uint64_t x, uint64_t y, uint64_t z) {
  writer.StartObject();
  writer.Key("x");
  writer.Uint64(x);
  writer.Key("y");
  writer.Uint64(y);
  writer.Key("z");
  writer.Uint64(z);
  writer.EndObject();
}

template <typename Writer>
void WriteCorrelation(Writer &writer, uint64_t corr, bool asynchronous) {
  writer.StartObject();
  writer.Key("internal");
  writer.Uint64(corr);
  writer.Key("external");
  writer.Uint64(0);
  if (!asynchronous) {
    writer.Key("ancestor");
    writer.Uint64(0);
  }
  writer.EndObject();
}

template <typename Writer>
void WriteAgent(Writer &writer, uint64_t handle, uint32_t kind,
                const Summary &summary) {
  const bool gpu = kind == 2;
  const uint64_t cu = gpu ? Positive(summary.cu_count, 1) : 0;
  const uint64_t wave = gpu ? Positive(summary.wave_front_size, 1) : 0;
  const uint64_t threads =
      gpu ? std::max<uint64_t>(wave, Positive(summary.threads_per_cu)) : 0;
  const uint64_t waves_per_cu = gpu ? std::max<uint64_t>(1, threads / wave) : 0;
  const uint64_t waves_per_simd =
      gpu ? std::max<uint64_t>(1, waves_per_cu / 4) : 0;
  const std::string name = gpu ? summary.device_name : ProcessorName();

  writer.StartObject();
  writer.Key("size");
  writer.Uint64(1);
  writer.Key("id");
  WriteHandle(writer, handle);
  writer.Key("type");
  writer.Uint(kind);
  writer.Key("cpu_cores_count");
  writer.Uint64(gpu ? 0 : CpuCount());
  writer.Key("simd_count");
  writer.Uint64(gpu ? cu * 4 : 0);
  writer.Key("mem_banks_count");
  writer.Uint64(0);
  writer.Key("caches_count");
  writer.Uint64(0);
  writer.Key("io_links_count");
  writer.Uint64(0);
  writer.Key("cpu_core_id_base");
  writer.Uint64(0);
  writer.Key("simd_id_base");
  writer.Uint64(0);
  writer.Key("max_waves_per_simd");
  writer.Uint64(waves_per_simd);
  writer.Key("lds_size_in_kb");
  writer.Uint64(gpu ? Positive(summary.local_mem_size) / 1024 : 0);
  writer.Key("gds_size_in_kb");
  writer.Uint64(0);
  writer.Key("num_gws");
  writer.Uint64(0);
  writer.Key("wave_front_size");
  writer.Uint64(wave);
  writer.Key("num_xcc");
  writer.Uint64(gpu ? 1 : 0);
  writer.Key("cu_count");
  writer.Uint64(cu);
  writer.Key("array_count");
  writer.Uint64(0);
  writer.Key("num_shader_banks");
  writer.Uint64(0);
  writer.Key("simd_arrays_per_engine");
  writer.Uint64(0);
  writer.Key("cu_per_simd_array");
  writer.Uint64(0);
  writer.Key("simd_per_cu");
  writer.Uint64(gpu ? 4 : 0);
  writer.Key("max_slots_scratch_cu");
  writer.Uint64(0);
  writer.Key("gfx_target_version");
  writer.Uint64(0);
  writer.Key("vendor_id");
  writer.Uint64(gpu ? 0x1002 : 0);
  writer.Key("device_id");
  writer.Uint64(0);
  writer.Key("location_id");
  writer.Uint64(0);
  writer.Key("domain");
  writer.Uint64(0);
  writer.Key("drm_render_minor");
  writer.Uint64(0);
  writer.Key("num_sdma_engines");
  writer.Uint64(0);
  writer.Key("num_sdma_xgmi_engines");
  writer.Uint64(0);
  writer.Key("num_sdma_queues_per_engine");
  writer.Uint64(0);
  writer.Key("num_cp_queues");
  writer.Uint64(0);
  writer.Key("max_engine_clk_ccompute");
  writer.Uint64(gpu ? Positive(summary.clock_khz) / 1000 : 0);
  writer.Key("max_engine_clk_fcompute");
  writer.Uint64(0);
  writer.Key("sdma_fw_version");
  writer.StartObject();
  writer.Key("uCodeSDMA");
  writer.Uint64(0);
  writer.Key("uCodeRes");
  writer.Uint64(0);
  writer.EndObject();
  writer.Key("fw_version");
  writer.StartObject();
  writer.Key("uCode");
  writer.Uint64(0);
  writer.Key("Major");
  writer.Uint64(0);
  writer.Key("Minor");
  writer.Uint64(0);
  writer.Key("Stepping");
  writer.Uint64(0);
  writer.EndObject();
  writer.Key("capability");
  writer.StartObject();
  for (const char *key : {"HotPluggable",
                          "HSAMMUPresent",
                          "SharedWithGraphics",
                          "QueueSizePowerOfTwo",
                          "QueueSize32bit",
                          "QueueIdleEvent",
                          "VALimit",
                          "WatchPointsSupported",
                          "WatchPointsTotalBits",
                          "DoorbellType",
                          "AQLQueueDoubleMap",
                          "DebugTrapSupported",
                          "WaveLaunchTrapOverrideSupported",
                          "WaveLaunchModeSupported",
                          "PreciseMemoryOperationsSupported",
                          "DEPRECATED_SRAM_EDCSupport",
                          "Mem_EDCSupport",
                          "RASEventNotify",
                          "ASICRevision",
                          "SRAM_EDCSupport",
                          "SVMAPISupported",
                          "CoherentHostAccess",
                          "DebugSupportedFirmware",
                          "Reserved"}) {
    writer.Key(key);
    writer.Uint64(0);
  }
  writer.EndObject();
  writer.Key("cu_per_engine");
  writer.Uint64(0);
  writer.Key("max_waves_per_cu");
  writer.Uint64(waves_per_cu);
  writer.Key("family_id");
  writer.Uint64(0);
  writer.Key("workgroup_max_size");
  writer.Uint64(gpu ? 1024 : 0);
  writer.Key("grid_max_size");
  writer.Uint64(0);
  writer.Key("local_mem_size");
  writer.Uint64(gpu ? Positive(summary.local_mem_size) : 0);
  writer.Key("hive_id");
  writer.Uint64(0);
  writer.Key("gpu_id");
  writer.Uint64(0);
  writer.Key("workgroup_max_dim");
  WriteDim(writer, gpu ? 1024 : 0, gpu ? 1024 : 0, gpu ? 1024 : 0);
  writer.Key("grid_max_dim");
  WriteDim(writer, 0, 0, 0);
  writer.Key("name");
  String(writer, name);
  writer.Key("vendor_name");
  String(writer, gpu ? "AMD" : "");
  writer.Key("product_name");
  String(writer, name);
  writer.Key("model_name");
  String(writer, name);
  writer.Key("node_id");
  writer.Uint64(handle - 1);
  writer.Key("logical_node_id");
  writer.Uint64(handle - 1);
  writer.Key("logical_node_type_id");
  writer.Uint64(0);
  writer.Key("runtime_visibility");
  writer.StartObject();
  writer.Key("hsa");
  writer.Bool(gpu);
  writer.Key("hip");
  writer.Bool(gpu);
  writer.Key("rccl");
  writer.Bool(gpu);
  writer.Key("rocdecode");
  writer.Bool(gpu);
  writer.EndObject();
  writer.Key("uuid");
  writer.StartObject();
  writer.Key("bytes");
  writer.StartArray();
  for (int i = 0; i < 16; ++i)
    writer.Uint(0);
  writer.EndArray();
  writer.EndObject();
  writer.Key("mem_banks");
  writer.StartArray();
  writer.EndArray();
  writer.Key("caches");
  writer.StartArray();
  writer.EndArray();
  writer.Key("io_links");
  writer.StartArray();
  writer.EndArray();
  writer.Key("gpu_index");
  writer.Int(gpu ? 0 : -1);
  writer.EndObject();
}

std::vector<std::string> AgentCsvRow(const Summary &summary, bool gpu) {
  std::vector<std::string> row(kAgentHeader.size(), "0");
  std::unordered_map<std::string, size_t> column;
  for (size_t i = 0; i < kAgentHeader.size(); ++i)
    column[kAgentHeader[i]] = i;
  auto set = [&](const char *key, const std::string &value) {
    row[column.at(key)] = value;
  };

  const uint64_t cu = gpu ? Positive(summary.cu_count, 1) : 0;
  const uint64_t wave = gpu ? Positive(summary.wave_front_size, 1) : 0;
  const uint64_t threads =
      gpu ? std::max<uint64_t>(wave, Positive(summary.threads_per_cu)) : 0;
  const uint64_t waves_per_cu = gpu ? std::max<uint64_t>(1, threads / wave) : 0;
  const uint64_t waves_per_simd =
      gpu ? std::max<uint64_t>(1, waves_per_cu / 4) : 0;
  const std::string name = gpu ? summary.device_name : ProcessorName();

  set("Node_Id", gpu ? "1" : "0");
  set("Logical_Node_Id", gpu ? "1" : "0");
  set("Agent_Type", gpu ? "GPU" : "CPU");
  set("Cpu_Cores_Count", U64(gpu ? 0 : CpuCount()));
  set("Simd_Count", U64(gpu ? cu * 4 : 0));
  set("Max_Waves_Per_Simd", U64(waves_per_simd));
  set("Lds_Size_In_Kb", U64(gpu ? Positive(summary.local_mem_size) / 1024 : 0));
  set("Wave_Front_Size", U64(wave));
  set("Num_Xcc", gpu ? "1" : "0");
  set("Cu_Count", U64(cu));
  set("Simd_Per_Cu", gpu ? "4" : "0");
  set("Vendor_Id", gpu ? "4098" : "0");
  set("Max_Engine_Clk_Ccompute",
      U64(gpu ? Positive(summary.clock_khz) / 1000 : 0));
  set("Max_Waves_Per_Cu", U64(waves_per_cu));
  set("Workgroup_Max_Size", gpu ? "1024" : "0");
  set("Local_Mem_Size", U64(gpu ? Positive(summary.local_mem_size) : 0));
  set("Workgroup_Max_Dim_X", gpu ? "1024" : "0");
  set("Workgroup_Max_Dim_Y", gpu ? "1024" : "0");
  set("Workgroup_Max_Dim_Z", gpu ? "1024" : "0");
  set("Name", name);
  set("Vendor_Name", gpu ? "AMD" : "");
  set("Product_Name", name);
  set("Model_Name", name);
  return row;
}

template <typename Writer>
void WriteKernelSymbol(Writer &writer, uint64_t kernel_id,
                       const std::string &name,
                       const KernelProperty &property) {
  writer.StartObject();
  writer.Key("size");
  writer.Uint64(1);
  writer.Key("kernel_id");
  writer.Uint64(kernel_id);
  writer.Key("code_object_id");
  writer.Uint64(0);
  writer.Key("kernel_name");
  String(writer, name);
  writer.Key("kernel_object");
  writer.Uint64(0);
  writer.Key("kernarg_segment_size");
  writer.Uint64(0);
  writer.Key("kernarg_segment_alignment");
  writer.Uint64(0);
  writer.Key("group_segment_size");
  writer.Uint64(property.static_lds);
  writer.Key("private_segment_size");
  writer.Uint64(property.scratch);
  writer.Key("sgpr_count");
  writer.Uint64(0);
  writer.Key("arch_vgpr_count");
  writer.Uint64(property.vgpr > 0 ? property.vgpr : 0);
  writer.Key("accum_vgpr_count");
  writer.Uint64(0);
  writer.Key("kernel_code_entry_byte_offset");
  writer.Int64(0);
  writer.Key("kernel_address");
  WriteHandle(writer, 0);
  writer.Key("formatted_kernel_name");
  String(writer, name);
  writer.Key("demangled_kernel_name");
  String(writer, name);
  writer.Key("truncated_kernel_name");
  String(writer, name);
  writer.EndObject();
}

template <typename Writer>
void WriteStringTables(
    Writer &writer,
    const std::map<uint32_t, std::map<uint32_t, std::string>> &api_names) {
  writer.StartArray();
  for (uint32_t kind = 0; kind <= kHipCompilerKind; ++kind) {
    std::string kind_name;
    std::vector<std::string> operations;
    if (kind == kMemoryCopyKind) {
      kind_name = "MEMORY_COPY";
      operations = {"", "MEMORY_COPY_HOST_TO_HOST",
                    "MEMORY_COPY_HOST_TO_DEVICE", "MEMORY_COPY_DEVICE_TO_HOST",
                    "MEMORY_COPY_DEVICE_TO_DEVICE"};
    } else if (kind == kKernelDispatchKind) {
      kind_name = "KERNEL_DISPATCH";
      operations = {"", "", "KERNEL_DISPATCH_COMPLETE"};
    } else if (kind == kMemoryAllocationKind) {
      kind_name = "MEMORY_ALLOCATION";
      operations = {"", "MEMORY_ALLOCATION_ALLOCATE",
                    "MEMORY_ALLOCATION_VMEM_ALLOCATE", "MEMORY_ALLOCATION_FREE",
                    "MEMORY_ALLOCATION_VMEM_FREE"};
    } else if (kind == kHipRuntimeKind || kind == kHipCompilerKind) {
      kind_name = kind == kHipRuntimeKind ? "HIP_RUNTIME_API_EXT"
                                          : "HIP_COMPILER_API_EXT";
      const auto found = api_names.find(kind);
      if (found != api_names.end() && !found->second.empty()) {
        operations.resize(found->second.rbegin()->first + 1);
        for (const auto &item : found->second)
          operations[item.first] = item.second;
      }
    }
    writer.StartObject();
    writer.Key("kind");
    String(writer, kind_name);
    writer.Key("operations");
    writer.StartArray();
    for (const auto &operation : operations)
      String(writer, operation);
    writer.EndArray();
    writer.EndObject();
  }
  writer.EndArray();
}

fs::path CreateTempDirectory(const fs::path &output_dir,
                             const std::string &prefix) {
  const uint64_t seed = GetTickCount64();
  for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
    const fs::path candidate =
        output_dir /
        ("." + prefix + "_" + std::to_string(GetCurrentProcessId()) + "_" +
         std::to_string(seed + attempt));
    std::error_code ec;
    if (fs::create_directory(candidate, ec))
      return candidate;
    if (ec && ec != std::errc::file_exists)
      throw std::runtime_error("cannot create temporary directory: " +
                               ec.message());
  }
  throw std::runtime_error("cannot allocate unique temporary directory");
}

void ReplaceFile(const fs::path &source, const fs::path &destination) {
  if (!MoveFileExW(source.c_str(), destination.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    const DWORD error = GetLastError();
    throw std::runtime_error("cannot publish " + destination.string() +
                             " (Win32 error " + std::to_string(error) + ")");
  }
}

std::map<std::string, fs::path> BasePaths(const fs::path &ops) {
  return {
      {"ops", ops},
      {"api", fs::path(ops.string() + ".api.csv")},
      {"kernels", fs::path(ops.string() + ".kernels.csv")},
      {"vram", fs::path(ops.string() + ".vram.csv")},
      {"roctx", fs::path(ops.string() + ".roctx.csv")},
      {"launch", fs::path(ops.string() + ".launch.csv")},
      {"summary", fs::path(ops.string() + ".summary.txt")},
  };
}

std::string InferPrefix(const fs::path &ops, uint64_t pid) {
  std::smatch match;
  const std::string name = ops.filename().string();
  if (std::regex_match(name, match, std::regex(R"(^\.(.+)_\d+_ops\.csv$)")))
    return match[1].str();
  return std::to_string(pid);
}

struct OpScan {
  uint64_t max_correlation = 0;
  std::vector<std::string> kernel_names;
};

OpScan ScanOps(const fs::path &ops) {
  CsvReader reader(ops);
  if (!reader.IsOpen())
    throw std::runtime_error("cannot read private ops spool: " + ops.string());
  CsvRow row;
  OpScan result;
  std::set<std::string> seen;
  while (reader.Next(row)) {
    if (Complete(row, {"Corr", "Begin_us", "End_us"}))
      result.max_correlation =
          std::max(result.max_correlation, ToU64(Get(row, "Corr")));
    if (Get(row, "Op") == "dispatch") {
      const std::string name = Get(row, "Kernel_Name");
      if (!name.empty() && seen.insert(name).second)
        result.kernel_names.push_back(name);
    }
  }
  return result;
}

std::map<std::string, fs::path>
ExportCapture(const fs::path &ops_path, const fs::path &output_dir_arg,
              std::string prefix, bool explicit_prefix, bool delete_spool) {
  const auto paths = BasePaths(ops_path);
  if (!fs::exists(paths.at("ops")))
    throw std::runtime_error("private ops spool not found: " +
                             ops_path.string());

  const fs::path output_dir = fs::absolute(output_dir_arg);
  fs::create_directories(output_dir);
  const Summary summary = ReadSummary(paths.at("summary"));
  uint64_t launch_pid = summary.pid;
  uint64_t resume_timestamp_ns = 0;
  if (fs::exists(paths.at("launch"))) {
    CsvReader launch(paths.at("launch"));
    CsvRow row;
    if (!launch.IsOpen() || !launch.Next(row))
      throw std::runtime_error("cannot read launch metadata spool");
    launch_pid = ToU64(Get(row, "Process_Id"), launch_pid);
    resume_timestamp_ns = ToU64(Get(row, "Resume_Timestamp_ns"));
  }
  if (!explicit_prefix)
    prefix = InferPrefix(ops_path, summary.pid);
  if (!explicit_prefix && fs::exists(output_dir / (prefix + "_results.json")))
    prefix += "_" + std::to_string(summary.pid);

  std::vector<CsvRow> property_rows;
  const auto properties =
      ReadKernelProperties(paths.at("kernels"), property_rows);
  const fs::path temp_dir = CreateTempDirectory(output_dir, prefix);

  const std::vector<std::string> output_names = {
      "agent_info.csv",        "hip_api_trace.csv",
      "kernel_trace.csv",      "kernel_trace_readable.csv",
      "memory_copy_trace.csv", "memory_allocation_trace.csv",
      "roctx_trace.csv",       "ktrace_launch_info.csv",
      "results.json",          "ktrace_extensions.json"};
  std::map<std::string, fs::path> final_out, out;
  for (const auto &name : output_names) {
    final_out[name] = output_dir / (prefix + "_" + name);
    out[name] = temp_dir / (prefix + "_" + name);
  }

  {
    CsvWriter agents(out.at("agent_info.csv"));
    if (!agents.IsOpen())
      throw std::runtime_error("cannot create agent CSV");
    agents.Header(kAgentHeader);
    agents.Row(AgentCsvRow(summary, false), {2, 49, 50, 51, 52});
    agents.Row(AgentCsvRow(summary, true), {2, 49, 50, 51, 52});
  }
  if (resume_timestamp_ns) {
    CsvWriter launch(out.at("ktrace_launch_info.csv"));
    if (!launch.IsOpen())
      throw std::runtime_error("cannot create launch metadata CSV");
    launch.Header({"Process_Id", "Resume_Timestamp"});
    launch.Row({U64(launch_pid), U64(resume_timestamp_ns)});
  }

  JsonFragment hip(temp_dir / "hip.frag");
  JsonFragment kernel(temp_dir / "kernel.frag");
  JsonFragment copy(temp_dir / "copy.frag");
  JsonFragment alloc(temp_dir / "alloc.frag");
  JsonFragment unmapped(temp_dir / "unmapped.frag");
  JsonFragment copy_extras(temp_dir / "copy_extras.frag");
  JsonFragment api_args(temp_dir / "api_args.frag", true);
  JsonFragment vram(temp_dir / "vram.frag");

  std::map<uint32_t, std::map<uint32_t, std::string>> api_names;
  std::unordered_map<uint64_t, uint64_t> stream_ids{{0, 0}};
  uint64_t next_stream = 1;
  const fs::path meta_path = temp_dir / "corr.bin";
  uint64_t max_corr = 0;
  uint64_t init_ns = std::numeric_limits<uint64_t>::max();
  uint64_t fini_ns = 0;
  uint64_t api_count = 0;
  uint64_t alloc_count = 0;

  {
    CsvWriter hip_csv(out.at("hip_api_trace.csv"));
    CsvWriter alloc_csv(out.at("memory_allocation_trace.csv"));
    if (!hip_csv.IsOpen() || !alloc_csv.IsOpen())
      throw std::runtime_error("cannot create API CSV output");
    hip_csv.Header(kHipHeader);
    alloc_csv.Header(kAllocHeader);

    std::fstream meta(meta_path, std::ios::binary | std::ios::in |
                                     std::ios::out | std::ios::trunc);
    if (!meta)
      throw std::runtime_error("cannot create correlation metadata");

    if (fs::exists(paths.at("api"))) {
      CsvReader reader(paths.at("api"));
      if (!reader.IsOpen())
        throw std::runtime_error("cannot read private API spool");
      CsvRow row;
      while (reader.Next(row)) {
        if (!Complete(row, {"Corr", "Api", "Api_Id", "Thread", "Enter_us",
                            "Exit_us"}))
          continue;
        const uint64_t corr = ToU64(Get(row, "Corr"));
        const uint32_t api_id =
            static_cast<uint32_t>(ToU64(Get(row, "Api_Id")));
        const uint32_t tid = static_cast<uint32_t>(ToU64(Get(row, "Thread")));
        const uint64_t start = Nanoseconds(Get(row, "Enter_us"));
        const uint64_t end = Nanoseconds(Get(row, "Exit_us"));
        init_ns = std::min(init_ns, start);
        fini_ns = std::max(fini_ns, end);
        const std::string function = Get(row, "Api");
        const uint32_t kind = function.rfind("__hip", 0) == 0 ? kHipCompilerKind
                                                              : kHipRuntimeKind;
        const std::string domain = kind == kHipCompilerKind
                                       ? "HIP_COMPILER_API_EXT"
                                       : "HIP_RUNTIME_API_EXT";
        api_names[kind][api_id] = function;
        hip_csv.Row({domain, function, U64(summary.pid), U64(tid), U64(corr),
                     U64(start), U64(end)},
                    {0, 1});
        ++api_count;

        const auto arguments = ParseArguments(Get(row, "Args"));
        const uint64_t raw_stream =
            arguments.count("stream") ? ToU64(arguments.at("stream")) : 0;
        auto stream_it = stream_ids.find(raw_stream);
        if (stream_it == stream_ids.end())
          stream_it = stream_ids.emplace(raw_stream, next_stream++).first;
        const uint64_t stream = stream_it->second;
        const auto grid =
            ParseDim(arguments.count("grid") ? arguments.at("grid") : "");
        const auto block =
            ParseDim(arguments.count("block") ? arguments.at("block") : "");
        const uint64_t shmem =
            arguments.count("shmem") ? ToU64(arguments.at("shmem")) : 0;

        CorrelationMeta record;
        record.api_id = api_id;
        record.tid = tid;
        record.stream = stream;
        record.gx = grid[0];
        record.gy = grid[1];
        record.gz = grid[2];
        record.bx = block[0];
        record.by = block[1];
        record.bz = block[2];
        record.dynamic_lds = shmem;
        if (corr > std::numeric_limits<uint64_t>::max() / sizeof(record))
          throw std::runtime_error("correlation id is too large");
        meta.seekp(static_cast<std::streamoff>(corr * sizeof(record)));
        meta.write(reinterpret_cast<const char *>(&record), sizeof(record));
        if (!meta)
          throw std::runtime_error("failed writing correlation metadata");
        max_corr = std::max(max_corr, corr);

        hip.Add(Json([&](auto &writer) {
          writer.StartObject();
          writer.Key("size");
          writer.Uint64(1);
          writer.Key("kind");
          writer.Uint(kind);
          writer.Key("operation");
          writer.Uint(api_id);
          writer.Key("correlation_id");
          WriteCorrelation(writer, corr, false);
          writer.Key("start_timestamp");
          writer.Uint64(start);
          writer.Key("end_timestamp");
          writer.Uint64(end);
          writer.Key("thread_id");
          writer.Uint64(tid);
          writer.Key("args");
          writer.StartArray();
          for (const auto &argument : arguments) {
            writer.StartObject();
            writer.Key("type");
            writer.String("uint64_t");
            writer.Key("name");
            String(writer, argument.first);
            writer.Key("value");
            String(writer, argument.second);
            writer.EndObject();
          }
          writer.EndArray();
          writer.Key("retval");
          writer.StartObject();
          writer.Key("uint64_t_retval");
          writer.Uint64(0);
          writer.EndObject();
          writer.Key("stream_id");
          WriteHandle(writer, stream);
          writer.EndObject();
        }));
        if (!arguments.empty())
          api_args.AddEntry(corr, Get(row, "Args"));

        const bool allocate_call =
            function == "hipMalloc" || function == "hipHostMalloc";
        const bool free_call =
            function == "hipFree" || function == "hipHostFree";
        if (allocate_call || free_call) {
          const bool host = function.find("Host") != std::string::npos;
          const uint32_t operation = allocate_call ? 1 : 3;
          const std::string operation_name = allocate_call
                                                 ? "MEMORY_ALLOCATION_ALLOCATE"
                                                 : "MEMORY_ALLOCATION_FREE";
          const std::string agent_display =
              allocate_call ? (host ? "Agent 0" : "Agent 1") : "";
          const uint64_t agent_handle = allocate_call ? (host ? 1 : 2) : 0;
          const uint64_t size = allocate_call && arguments.count("size")
                                    ? ToU64(arguments.at("size"))
                                    : 0;
          const uint64_t address =
              arguments.count("ptr") ? ToU64(arguments.at("ptr")) : 0;
          const std::string address_text = HexAddress(address);
          alloc_csv.Row({"MEMORY_ALLOCATION", operation_name, agent_display,
                         U64(size), address_text, U64(corr), U64(start),
                         U64(end)},
                        {0, 1, 2, 4});
          alloc.Add(Json([&](auto &writer) {
            writer.StartObject();
            writer.Key("size");
            writer.Uint64(1);
            writer.Key("kind");
            writer.Uint(kMemoryAllocationKind);
            writer.Key("operation");
            writer.Uint(operation);
            writer.Key("thread_id");
            writer.Uint64(tid);
            writer.Key("correlation_id");
            WriteCorrelation(writer, corr, false);
            writer.Key("start_timestamp");
            writer.Uint64(start);
            writer.Key("end_timestamp");
            writer.Uint64(end);
            writer.Key("agent_id");
            WriteHandle(writer, agent_handle);
            writer.Key("address");
            String(writer, address_text);
            writer.Key("allocation_size");
            writer.Uint64(size);
            writer.Key("stream_id");
            WriteHandle(writer, stream);
            writer.EndObject();
          }));
          ++alloc_count;
        }
      }
    }
  }
  if (!api_count)
    fs::remove(out.at("hip_api_trace.csv"));
  if (!alloc_count)
    fs::remove(out.at("memory_allocation_trace.csv"));

  uint64_t roctx_count = 0;
  if (fs::exists(paths.at("roctx"))) {
    CsvReader reader(paths.at("roctx"));
    CsvWriter writer(out.at("roctx_trace.csv"));
    if (!reader.IsOpen() || !writer.IsOpen())
      throw std::runtime_error("cannot process ROCTx spool");
    writer.Header(kRoctxHeader);
    CsvRow row;
    while (reader.Next(row)) {
      const uint64_t timestamp_ns = static_cast<uint64_t>(
          std::llround(ToDouble(Get(row, "Timestamp_us")) * 1000.0));
      writer.Row(
          {
              "ROCTX",
              Get(row, "Op"),
              Get(row, "Process"),
              Get(row, "Thread"),
              Get(row, "Range_Id"),
              std::to_string(timestamp_ns),
              Get(row, "Message"),
          },
          {0, 1, 6});
      ++roctx_count;
    }
  }
  if (!roctx_count)
    fs::remove(out.at("roctx_trace.csv"));

  const OpScan op_scan = ScanOps(paths.at("ops"));
  const auto readable_names = ResolveKernelNames(op_scan.kernel_names);
  const uint64_t max_op_corr = op_scan.max_correlation;
  const uint64_t max_stored_corr = std::max(max_corr, max_op_corr);
  if (max_stored_corr >
      (std::numeric_limits<uint64_t>::max() / sizeof(CorrelationMeta)) - 1)
    throw std::runtime_error("correlation metadata file would overflow");
  fs::resize_file(
      meta_path,
      std::max<uint64_t>(1, (max_stored_corr + 1) * sizeof(CorrelationMeta)));

  std::map<std::string, uint64_t> kernel_ids;
  uint64_t dispatch_id = 0;
  uint64_t copy_count = 0;
  uint64_t next_synthetic_corr = max_corr + 1;
  {
    CsvWriter kernel_csv(out.at("kernel_trace.csv"));
    CsvWriter readable_kernel_csv(out.at("kernel_trace_readable.csv"));
    CsvWriter copy_csv(out.at("memory_copy_trace.csv"));
    if (!kernel_csv.IsOpen() || !readable_kernel_csv.IsOpen() ||
        !copy_csv.IsOpen())
      throw std::runtime_error("cannot create GPU CSV output");
    kernel_csv.Header(kKernelHeader);
    readable_kernel_csv.Header(kKernelHeader);
    copy_csv.Header(kCopyHeader);

    std::ifstream meta(meta_path, std::ios::binary);
    CsvReader reader(paths.at("ops"));
    if (!meta || !reader.IsOpen())
      throw std::runtime_error("cannot process private ops spool");
    CsvRow row;
    while (reader.Next(row)) {
      if (!Complete(row, {"Op", "Kind", "Bytes", "Device", "Queue", "Corr",
                          "Begin_us", "End_us"}))
        continue;
      const uint64_t raw_corr = ToU64(Get(row, "Corr"));
      const uint64_t corr = raw_corr ? raw_corr : next_synthetic_corr++;
      const uint64_t start = Nanoseconds(Get(row, "Begin_us"));
      const uint64_t end = Nanoseconds(Get(row, "End_us"));
      init_ns = std::min(init_ns, start);
      fini_ns = std::max(fini_ns, end);

      CorrelationMeta correlation;
      if (raw_corr && raw_corr <= max_stored_corr) {
        meta.clear();
        meta.seekg(static_cast<std::streamoff>(raw_corr * sizeof(correlation)));
        meta.read(reinterpret_cast<char *>(&correlation), sizeof(correlation));
        if (meta.gcount() != sizeof(correlation))
          correlation = {};
      }
      uint64_t tid = correlation.tid ? correlation.tid : summary.pid;
      const uint64_t queue = ToU64(Get(row, "Queue"));
      const uint64_t bytes = ToU64(Get(row, "Bytes"));

      if (Get(row, "Op") == "dispatch") {
        ++dispatch_id;
        const std::string name = Get(row, "Kernel_Name");
        auto kernel_it = kernel_ids.find(name);
        if (kernel_it == kernel_ids.end())
          kernel_it = kernel_ids.emplace(name, kernel_ids.size() + 1).first;
        const uint64_t kernel_id = kernel_it->second;
        KernelProperty property;
        const auto prop_it = properties.find(name);
        if (prop_it != properties.end())
          property = prop_it->second;
        const uint64_t total_lds =
            property.static_lds + correlation.dynamic_lds;
        const uint64_t lds_block =
            property.static_lds ? ((property.static_lds + 511) & ~uint64_t{511})
                                : 0;
        std::vector<std::string> kernel_row{"KERNEL_DISPATCH",
                                            "Agent 1",
                                            U64(queue),
                                            U64(correlation.stream),
                                            U64(tid),
                                            U64(dispatch_id),
                                            U64(kernel_id),
                                            name,
                                            U64(corr),
                                            U64(start),
                                            U64(end),
                                            U64(lds_block),
                                            U64(property.scratch),
                                            I64(property.vgpr),
                                            "0",
                                            "0",
                                            U64(correlation.bx),
                                            U64(correlation.by),
                                            U64(correlation.bz),
                                            U64(correlation.gx),
                                            U64(correlation.gy),
                                            U64(correlation.gz)};
        kernel_csv.Row(kernel_row, {0, 1, 7});
        const auto readable = readable_names.find(name);
        if (readable != readable_names.end())
          kernel_row[7] = readable->second;
        readable_kernel_csv.Row(kernel_row, {0, 1, 7});

        kernel.Add(Json([&](auto &writer) {
          writer.StartObject();
          writer.Key("size");
          writer.Uint64(1);
          writer.Key("kind");
          writer.Uint(kKernelDispatchKind);
          writer.Key("operation");
          writer.Uint(2);
          writer.Key("thread_id");
          writer.Uint64(tid);
          writer.Key("correlation_id");
          WriteCorrelation(writer, corr, true);
          writer.Key("start_timestamp");
          writer.Uint64(start);
          writer.Key("end_timestamp");
          writer.Uint64(end);
          writer.Key("dispatch_info");
          writer.StartObject();
          writer.Key("size");
          writer.Uint64(1);
          writer.Key("agent_id");
          WriteHandle(writer, 2);
          writer.Key("queue_id");
          WriteHandle(writer, queue);
          writer.Key("kernel_id");
          writer.Uint64(kernel_id);
          writer.Key("dispatch_id");
          writer.Uint64(dispatch_id);
          writer.Key("private_segment_size");
          writer.Uint64(property.scratch);
          writer.Key("group_segment_size");
          writer.Uint64(total_lds);
          writer.Key("workgroup_size");
          WriteDim(writer, correlation.bx, correlation.by, correlation.bz);
          writer.Key("grid_size");
          WriteDim(writer, correlation.gx, correlation.gy, correlation.gz);
          writer.EndObject();
          writer.Key("stream_id");
          WriteHandle(writer, correlation.stream);
          writer.EndObject();
        }));
      } else {
        const auto direction = kCopyDirections.find(Get(row, "Kind"));
        if (Get(row, "Op") == "copy" && direction != kCopyDirections.end()) {
          const auto &d = direction->second;
          copy_csv.Row({"MEMORY_COPY", d.name, U64(correlation.stream),
                        "Agent " + U64(d.source_agent - 1),
                        "Agent " + U64(d.destination_agent - 1), U64(corr),
                        U64(start), U64(end)},
                       {0, 1, 3, 4});
          copy.Add(Json([&](auto &writer) {
            writer.StartObject();
            writer.Key("size");
            writer.Uint64(1);
            writer.Key("kind");
            writer.Uint(kMemoryCopyKind);
            writer.Key("operation");
            writer.Uint(d.operation);
            writer.Key("thread_id");
            writer.Uint64(tid);
            writer.Key("correlation_id");
            WriteCorrelation(writer, corr, true);
            writer.Key("start_timestamp");
            writer.Uint64(start);
            writer.Key("end_timestamp");
            writer.Uint64(end);
            writer.Key("dst_agent_id");
            WriteHandle(writer, d.destination_agent);
            writer.Key("src_agent_id");
            WriteHandle(writer, d.source_agent);
            writer.Key("bytes");
            writer.Uint64(bytes);
            writer.Key("stream_id");
            WriteHandle(writer, correlation.stream);
            writer.EndObject();
          }));
          copy_extras.Add(Json([&](auto &writer) {
            writer.StartObject();
            writer.Key("correlation_id");
            writer.Uint64(corr);
            writer.Key("start_timestamp");
            writer.Uint64(start);
            writer.Key("bytes");
            writer.Uint64(bytes);
            writer.Key("device");
            writer.String("Agent 1");
            writer.Key("queue");
            writer.Uint64(queue);
            writer.Key("stream");
            writer.Uint64(correlation.stream);
            writer.EndObject();
          }));
          ++copy_count;
        } else {
          unmapped.Add(Json([&](auto &writer) {
            writer.StartObject();
            writer.Key("op");
            String(writer, Get(row, "Op"));
            writer.Key("kind");
            String(writer, Get(row, "Kind"));
            writer.Key("bytes");
            writer.Uint64(bytes);
            writer.Key("device");
            writer.String("Agent 1");
            writer.Key("queue");
            writer.Uint64(queue);
            writer.Key("correlation_id");
            writer.Uint64(corr);
            writer.Key("start_timestamp");
            writer.Uint64(start);
            writer.Key("end_timestamp");
            writer.Uint64(end);
            writer.EndObject();
          }));
        }
      }
    }
  }
  if (!dispatch_id) {
    fs::remove(out.at("kernel_trace.csv"));
    fs::remove(out.at("kernel_trace_readable.csv"));
  }
  if (!copy_count)
    fs::remove(out.at("memory_copy_trace.csv"));

  if (fs::exists(paths.at("vram"))) {
    CsvReader reader(paths.at("vram"));
    if (!reader.IsOpen())
      throw std::runtime_error("cannot read private VRAM spool");
    CsvRow row;
    while (reader.Next(row)) {
      if (!Complete(row,
                    {"Time_us", "Free_Bytes", "Total_Bytes", "Used_Bytes"}))
        continue;
      vram.Add(Json([&](auto &writer) {
        writer.StartObject();
        writer.Key("timestamp");
        writer.Uint64(Nanoseconds(Get(row, "Time_us")));
        writer.Key("free_bytes");
        writer.Uint64(ToU64(Get(row, "Free_Bytes")));
        writer.Key("total_bytes");
        writer.Uint64(ToU64(Get(row, "Total_Bytes")));
        writer.Key("used_bytes");
        writer.Uint64(ToU64(Get(row, "Used_Bytes")));
        writer.EndObject();
      }));
    }
  }

  hip.Close();
  kernel.Close();
  copy.Close();
  alloc.Close();
  unmapped.Close();
  copy_extras.Close();
  api_args.Close();
  vram.Close();

  {
    const auto [windows_release, windows_version] = WindowsVersion();
    std::vector<char> result_buffer(1024 * 1024);
    std::ofstream result;
    result.rdbuf()->pubsetbuf(
        result_buffer.data(),
        static_cast<std::streamsize>(result_buffer.size()));
    result.open(out.at("results.json"), std::ios::binary);
    if (!result)
      throw std::runtime_error("cannot create results JSON");
    result << "{\"rocprofiler-sdk-tool\":[{";
    result << "\"metadata\":";
    result << Json([&](auto &writer) {
      writer.StartObject();
      writer.Key("node");
      writer.StartObject();
      writer.Key("id");
      writer.Uint64(0);
      writer.Key("hash");
      writer.Uint64(0);
      writer.Key("machine_id");
      writer.String("");
      writer.Key("system_name");
      writer.String("Windows");
      writer.Key("hostname");
      String(writer, Hostname());
      writer.Key("release");
      String(writer, windows_release);
      writer.Key("version");
      String(writer, windows_version);
      writer.Key("hardware_name");
      String(writer, MachineName());
      writer.Key("domain_name");
      writer.String("");
      writer.EndObject();
      writer.Key("pid");
      writer.Uint64(summary.pid);
      writer.Key("init_time");
      writer.Uint64(resume_timestamp_ns
                        ? resume_timestamp_ns
                        : (init_ns == std::numeric_limits<uint64_t>::max()
                               ? 0
                               : init_ns));
      writer.Key("fini_time");
      writer.Uint64(fini_ns);
      writer.Key("command");
      writer.StartArray();
      writer.EndArray();
      writer.Key("config");
      writer.StartObject();
      writer.Key("output_path");
      String(writer, AbsoluteUtf8(output_dir));
      writer.Key("output_file");
      String(writer, prefix);
      writer.Key("tmp_directory");
      String(writer, AbsoluteUtf8(temp_dir));
      writer.Key("raw_output_path");
      String(writer, AbsoluteUtf8(ops_path.parent_path()));
      writer.Key("raw_output_file");
      String(writer, ops_path.filename().u8string());
      writer.Key("raw_tmp_directory");
      String(writer, AbsoluteUtf8(temp_dir));
      writer.Key("perfetto_shmem_size_hint");
      writer.Uint64(0);
      writer.Key("perfetto_buffer_size");
      writer.Uint64(0);
      writer.Key("perfetto_buffer_fill_policy");
      writer.String("");
      writer.Key("perfetto_backend");
      writer.String("");
      writer.Key("summary");
      writer.Bool(false);
      writer.Key("summary_per_domain");
      writer.Bool(false);
      writer.Key("summary_groups");
      writer.StartArray();
      writer.EndArray();
      writer.Key("summary_unit");
      writer.String("nsec");
      writer.Key("summary_file");
      writer.String("");
      writer.Key("csv_output");
      writer.Bool(true);
      writer.Key("json_output");
      writer.Bool(true);
      writer.Key("pftrace_output");
      writer.Bool(false);
      writer.Key("otf2_output");
      writer.Bool(false);
      writer.Key("summary_output");
      writer.Bool(false);
      writer.Key("rocpd_output");
      writer.Bool(false);
      writer.Key("kernel_rename");
      writer.Bool(false);
      writer.Key("group_by_queue");
      writer.Bool(false);
      writer.EndObject();
      writer.EndObject();
    });
    result << ",\"agents\":";
    result << Json([&](auto &writer) {
      writer.StartArray();
      WriteAgent(writer, 1, 1, summary);
      WriteAgent(writer, 2, 2, summary);
      writer.EndArray();
    });
    result << ",\"counters\":[],\"strings\":";
    result << Json([&](auto &writer) {
      writer.StartObject();
      writer.Key("callback_records");
      writer.StartArray();
      writer.EndArray();
      writer.Key("buffer_records");
      WriteStringTables(writer, api_names);
      writer.Key("marker_api");
      writer.StartArray();
      writer.EndArray();
      writer.Key("correlation_id");
      writer.StartObject();
      writer.Key("external");
      writer.StartArray();
      writer.EndArray();
      writer.EndObject();
      writer.Key("counters");
      writer.StartObject();
      writer.Key("dimension_ids");
      writer.StartArray();
      writer.EndArray();
      writer.EndObject();
      writer.Key("pc_sample_instructions");
      writer.StartArray();
      writer.EndArray();
      writer.Key("pc_sample_comments");
      writer.StartArray();
      writer.EndArray();
      writer.Key("att_filenames");
      writer.StartArray();
      writer.EndArray();
      writer.Key("code_object_snapshot_filenames");
      writer.StartArray();
      writer.EndArray();
      writer.EndObject();
    });
    result << ",\"code_objects\":[],\"kernel_symbols\":";
    result << Json([&](auto &writer) {
      writer.StartArray();
      WriteKernelSymbol(writer, 0, "<unknown>", KernelProperty{});
      std::vector<std::pair<std::string, uint64_t>> by_id(kernel_ids.begin(),
                                                          kernel_ids.end());
      std::sort(by_id.begin(), by_id.end(),
                [](const auto &left, const auto &right) {
                  return left.second < right.second;
                });
      for (const auto &item : by_id) {
        KernelProperty property;
        const auto found = properties.find(item.first);
        if (found != properties.end())
          property = found->second;
        WriteKernelSymbol(writer, item.second, item.first, property);
      }
      writer.EndArray();
    });
    result << ",\"host_functions\":[],\"summary\":[]";
    result << ",\"callback_records\":{\"counter_collection\":[]}";
    result << ",\"buffer_records\":{";
    result << "\"kernel_dispatch\":";
    kernel.CopyTo(result);
    result << ",\"hip_api\":";
    hip.CopyTo(result);
    result << ",\"hsa_api\":[],\"marker_api\":[],\"rccl_api\":[]";
    result << ",\"memory_copy\":";
    copy.CopyTo(result);
    result << ",\"memory_allocation\":";
    alloc.CopyTo(result);
    result << ",\"scratch_memory\":[],\"rocdecode_api\":[],\"rocjpeg_api\":[]";
    result << ",\"pc_sample_host_trap\":[],\"pc_sample_stochastic\":[]";
    result << "}}]}\n";
    if (!result)
      throw std::runtime_error("failed writing results JSON");
  }

  {
    std::vector<char> extension_buffer(1024 * 1024);
    std::ofstream extension;
    extension.rdbuf()->pubsetbuf(
        extension_buffer.data(),
        static_cast<std::streamsize>(extension_buffer.size()));
    extension.open(out.at("ktrace_extensions.json"), std::ios::binary);
    if (!extension)
      throw std::runtime_error("cannot create ktrace extensions JSON");
    extension << '{';
    extension << "\"schema\":\"ktrace-extensions-1.1\",";
    extension
        << "\"rocprofiler_schema\":\"ROCm 7.2.1 / ROCprofiler-SDK 1.1.0\",";
    extension << "\"source_spool\":" << JsonEscape(AbsoluteUtf8(ops_path))
              << ',';
    extension << "\"process_resume_timestamp\":" << resume_timestamp_ns << ',';
    extension
        << "\"field_quality\":{"
           "\"timestamps\":\"exact_qpc_nanoseconds\","
           "\"queue_id\":\"exact_rocclr_hardware_queue\","
           "\"correlation_id\":\"exact_hip_runtime_when_present_else_synthetic_"
           "gpu_record_id\","
           "\"stream_id\":\"synthetic_session_id_from_hip_stream_pointer\","
           "\"thread_id\":\"exact_from_correlated_hip_api_else_synthetic_"
           "process_id\","
           "\"agent_id\":\"synthetic_cpu_gpu_topology\","
           "\"dispatch_id\":\"synthetic_monotonic\","
           "\"kernel_id\":\"synthetic_by_kernel_name\","
           "\"sgpr_count\":\"unavailable_zero\","
           "\"accum_vgpr_count\":\"unavailable_zero\","
           "\"abi_size_fields\":\"synthetic_nonzero_windows_records_are_not_"
           "sdk_structs\","
           "\"memory_allocation\":\"hip_api_level_not_hsa_allocation_trace\"},";
    extension << "\"kernel_properties\":";
    extension << Json([&](auto &writer) {
      writer.StartArray();
      for (const auto &row : property_rows) {
        writer.StartObject();
        std::vector<std::pair<std::string, std::string>> fields(row.begin(),
                                                                row.end());
        std::sort(fields.begin(), fields.end());
        for (const auto &field : fields) {
          String(writer, field.first);
          String(writer, field.second);
        }
        writer.EndObject();
      }
      writer.EndArray();
    });
    extension << ",\"vram_samples\":";
    vram.CopyTo(extension);
    extension << ",\"capture_summary\":" << JsonEscape(summary.text);
    extension << ",\"memory_copy_extras\":";
    copy_extras.CopyTo(extension);
    extension << ",\"unmapped_gpu_commands\":";
    unmapped.CopyTo(extension);
    extension << ",\"api_arguments\":";
    api_args.CopyTo(extension);
    extension << "}\n";
    if (!extension)
      throw std::runtime_error("failed writing ktrace extensions JSON");
  }

  for (JsonFragment *fragment : {&hip, &kernel, &copy, &alloc, &unmapped,
                                 &copy_extras, &api_args, &vram}) {
    std::error_code ec;
    fs::remove(fragment->Path(), ec);
  }
  {
    std::error_code ec;
    fs::remove(meta_path, ec);
  }

  for (const auto &name : output_names) {
    if (fs::exists(out.at(name)))
      ReplaceFile(out.at(name), final_out.at(name));
    else {
      std::error_code ec;
      fs::remove(final_out.at(name), ec);
      if (ec)
        throw std::runtime_error("cannot remove stale output " +
                                 final_out.at(name).string());
    }
  }
  {
    std::error_code ec;
    fs::remove(temp_dir, ec);
  }

  if (delete_spool) {
    for (const auto &item : paths) {
      std::error_code ec;
      fs::remove(item.second, ec);
      if (ec)
        throw std::runtime_error("output published but cannot delete spool " +
                                 item.second.string());
    }
  }
  return final_out;
}

bool PrivateSpoolName(const fs::path &path) {
  const std::string name = path.filename().string();
  return name.size() > 9 && name.front() == '.' &&
         name.compare(name.size() - 8, 8, "_ops.csv") == 0;
}

std::string PrivatePrefix(const fs::path &spool) {
  std::smatch match;
  const std::string name = spool.filename().string();
  return std::regex_match(name, match, std::regex(R"(^\.(.+)_\d+_ops\.csv$)"))
             ? match[1].str()
             : std::string{};
}

void FinalizeDirectory(const fs::path &directory_arg) {
  const fs::path directory = fs::absolute(directory_arg);
  if (!fs::is_directory(directory))
    throw std::runtime_error("capture directory not found: " +
                             directory.string());
  std::vector<fs::path> spools;
  for (const auto &entry : fs::directory_iterator(directory))
    if (entry.is_regular_file() && PrivateSpoolName(entry.path()))
      spools.push_back(entry.path());
  std::sort(spools.begin(), spools.end());
  if (spools.empty())
    throw std::runtime_error("no private ktrace spool found in " +
                             directory.string());

  std::vector<std::string> private_prefixes;
  for (const auto &spool : spools) {
    const std::string prefix = PrivatePrefix(spool);
    if (!prefix.empty())
      private_prefixes.push_back(prefix);
    std::cout << "[finalize] " << spool.filename().string() << '\n';
    ExportCapture(spool, directory, {}, false, true);
  }

  for (const auto &entry : fs::directory_iterator(directory)) {
    if (!entry.is_directory())
      continue;
    const std::string name = entry.path().filename().string();
    for (const auto &prefix : private_prefixes) {
      if (name.rfind("." + prefix + "_", 0) == 0) {
        std::error_code ec;
        fs::remove_all(entry.path(), ec);
        break;
      }
    }
  }
}

void PrintFinalizeHelp() {
  std::cout << "usage:\n"
            << "  ktrace_postprocess finalize <capture-dir>\n"
            << "  ktrace_postprocess finalize <ops.csv> --output-dir <dir>\n"
            << "      [--output-file <prefix>] [--delete-spool]\n";
}
} // namespace

int FinalizeCommand(const std::vector<std::string> &args) {
  if (args.empty() || args[0] == "-h" || args[0] == "--help" ||
      args[0] == "help") {
    PrintFinalizeHelp();
    return 0;
  }

  const fs::path source = args[0];
  if (fs::is_directory(source)) {
    if (args.size() != 1)
      throw std::runtime_error("directory finalize accepts only <capture-dir>");
    FinalizeDirectory(source);
    return 0;
  }

  fs::path output_dir;
  std::string prefix;
  bool explicit_prefix = false;
  bool delete_spool = false;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--output-dir") {
      if (++i == args.size())
        throw std::runtime_error("--output-dir requires a value");
      output_dir = args[i];
    } else if (args[i] == "--output-file") {
      if (++i == args.size())
        throw std::runtime_error("--output-file requires a value");
      prefix = args[i];
      explicit_prefix = true;
    } else if (args[i] == "--delete-spool")
      delete_spool = true;
    else
      throw std::runtime_error("unknown finalize option: " + args[i]);
  }
  if (output_dir.empty())
    throw std::runtime_error(
        "--output-dir is required when finalizing an ops spool");

  const auto outputs =
      ExportCapture(source, output_dir, prefix, explicit_prefix, delete_spool);
  for (const auto &item : outputs)
    if (fs::exists(item.second))
      std::cout << "-> " << item.second.string() << " (" << std::fixed
                << std::setprecision(1)
                << static_cast<double>(fs::file_size(item.second)) / 1.0e6
                << " MB)\n";
  return 0;
}
} // namespace ktrace
