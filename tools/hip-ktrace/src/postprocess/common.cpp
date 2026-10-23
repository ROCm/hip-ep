/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"
#include "amd_comgr.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ktrace {
namespace {
const std::vector<std::string> kDomainSuffixes = {
    "_kernel_trace_readable.csv",
    "_kernel_trace.csv",
    "_memory_copy_trace.csv",
    "_memory_allocation_trace.csv",
    "_hip_api_trace.csv",
    "_roctx_trace.csv",
    "_agent_info.csv",
    "_ktrace_launch_info.csv",
    "_hip_runtime_info.csv",
    "_cpu_sched.csv",
    "_cpu_samples.csv",
    "_cpu_images.csv",
    "_results.json",
    "_ktrace_extensions.json"};

std::string Env(const char *name) {
  const DWORD n = GetEnvironmentVariableA(name, nullptr, 0);
  if (!n)
    return {};
  std::string value(n - 1, '\0');
  GetEnvironmentVariableA(name, value.data(), n);
  return value;
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

std::string KernelLabel(std::string value) {
  const std::string anon = "(anonymous namespace)::";
  for (size_t p = value.find(anon); p != std::string::npos;
       p = value.find(anon))
    value.erase(p, anon.size());

  int depth = 0;
  size_t cut = value.size();
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '<')
      ++depth;
    else if (value[i] == '>')
      --depth;
    else if (value[i] == '(' && depth == 0) {
      cut = i;
      break;
    }
  }
  value.resize(cut);
  depth = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '<')
      ++depth;
    else if (value[i] == '>')
      --depth;
    else if (value[i] == ' ' && depth == 0) {
      value = value.substr(i + 1);
      break;
    }
  }
  value = std::regex_replace(value, std::regex(",\\s+"), ",");
  value = Trim(value);
  std::smatch match;
  if (std::regex_search(
          value, match,
          std::regex(R"(^(Cijk_\w*?_MT\d+x\d+x\d+_MI\d+x\d+x\d+)_.*)")))
    return match[1].str();
  return value;
}

std::string KernelFallback(const std::string &mangled) {
  std::smatch match;
  if (std::regex_match(mangled, match, std::regex(R"(^_Z(\d+)(.*)$)"))) {
    const size_t n = static_cast<size_t>(std::stoul(match[1].str()));
    return match[2].str().substr(0, n);
  }
  if (!std::regex_match(mangled, match, std::regex(R"(^_ZN(.*)$)")))
    return mangled;
  std::string rest = match[1].str(), last = mangled;
  while (std::regex_match(rest, match, std::regex(R"(^(\d+)(.*)$)"))) {
    const size_t n = static_cast<size_t>(std::stoul(match[1].str()));
    last = match[2].str().substr(0, n);
    rest = match[2].str().substr(n);
  }
  return last;
}

std::set<std::string> SplitTokens(const std::string &raw) {
  std::set<std::string> out;
  size_t begin = 0;
  while (begin <= raw.size()) {
    const size_t end = raw.find('_', begin);
    out.insert(raw.substr(begin, end == std::string::npos ? end : end - begin));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return out;
}
} // namespace

std::vector<std::string> ParseCsvLine(const std::string &line) {
  std::vector<std::string> out;
  std::string field;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else
        quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      out.push_back(std::move(field));
      field.clear();
    } else if (ch != '\r' && ch != '\n')
      field.push_back(ch);
  }
  out.push_back(std::move(field));
  return out;
}

CsvReader::CsvReader(const fs::path &path) : buffer_(1024 * 1024) {
  file_.rdbuf()->pubsetbuf(buffer_.data(),
                           static_cast<std::streamsize>(buffer_.size()));
  file_.open(path, std::ios::binary);
  std::string line;
  if (std::getline(file_, line)) {
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF)
      line.erase(0, 3);
    header_ = ParseCsvLine(line);
  }
}

bool CsvReader::IsOpen() const { return file_.is_open() && !header_.empty(); }

bool CsvReader::Next(CsvRow &row) {
  std::string line;
  if (!std::getline(file_, line))
    return false;
  auto values = ParseCsvLine(line);
  row.clear();
  for (size_t i = 0; i < header_.size(); ++i)
    row.emplace(header_[i],
                i < values.size() ? std::move(values[i]) : std::string{});
  return true;
}

CsvWriter::CsvWriter(const fs::path &path) : buffer_(1024 * 1024) {
  file_.rdbuf()->pubsetbuf(buffer_.data(),
                           static_cast<std::streamsize>(buffer_.size()));
  file_.open(path, std::ios::binary);
}
bool CsvWriter::IsOpen() const { return file_.is_open(); }

void CsvWriter::Header(const std::vector<std::string> &values) {
  std::vector<size_t> columns(values.size());
  for (size_t i = 0; i < columns.size(); ++i)
    columns[i] = i;
  Row(values, columns);
}

void CsvWriter::Row(const std::vector<std::string> &values,
                    const std::vector<size_t> &string_columns) {
  std::set<size_t> strings(string_columns.begin(), string_columns.end());
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      file_ << ',';
    file_ << (strings.count(i) ? CsvQuote(values[i]) : values[i]);
  }
  file_ << '\n';
}

std::string CsvQuote(const std::string &value) { return '"' + value + '"'; }

std::string JsonEscape(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec;
      else
        out << static_cast<char>(ch);
    }
  }
  out << '"';
  return out.str();
}

std::string Lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool EndsWithInsensitive(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         Lower(value.substr(value.size() - suffix.size())) == Lower(suffix);
}

uint64_t ToU64(const std::string &value, uint64_t fallback) {
  if (value.empty())
    return fallback;
  char *end = nullptr;
  const unsigned long long result = std::strtoull(value.c_str(), &end, 0);
  return end != value.c_str() ? static_cast<uint64_t>(result) : fallback;
}

int64_t ToI64(const std::string &value, int64_t fallback) {
  if (value.empty())
    return fallback;
  char *end = nullptr;
  const long long result = std::strtoll(value.c_str(), &end, 0);
  return end != value.c_str() ? static_cast<int64_t>(result) : fallback;
}

double ToDouble(const std::string &value, double fallback) {
  if (value.empty())
    return fallback;
  char *end = nullptr;
  const double result = std::strtod(value.c_str(), &end);
  return end != value.c_str() ? result : fallback;
}

std::string HumanBytes(double bytes) {
  static const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  size_t i = 0;
  while (std::abs(bytes) >= 1024.0 && i < 4) {
    bytes /= 1024.0;
    ++i;
  }
  std::ostringstream out;
  if (i == 0)
    out << static_cast<int64_t>(bytes) << " B";
  else
    out << std::fixed << std::setprecision(1) << bytes << ' ' << units[i];
  return out.str();
}

std::string NowIso8601() {
  SYSTEMTIME st{};
  GetLocalTime(&st);
  TIME_ZONE_INFORMATION tz{};
  const DWORD mode = GetTimeZoneInformation(&tz);
  LONG bias = tz.Bias + (mode == TIME_ZONE_ID_DAYLIGHT   ? tz.DaylightBias
                         : mode == TIME_ZONE_ID_STANDARD ? tz.StandardBias
                                                         : 0);
  const char sign = bias <= 0 ? '+' : '-';
  bias = std::abs(bias);
  char text[64];
  std::snprintf(text, sizeof(text),
                "%04u-%02u-%02uT%02u:%02u:%02u%c%02ld:%02ld", st.wYear,
                st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, sign,
                bias / 60, bias % 60);
  return text;
}

std::string Hostname() {
  char name[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD size = sizeof(name);
  return GetComputerNameA(name, &size) ? std::string(name, size)
                                       : std::string{};
}

std::string AbsoluteUtf8(const fs::path &path) {
  return fs::absolute(path).u8string();
}

fs::path Capture::Domain(const std::string &domain) const {
  if (IsLegacy()) {
    if (domain == "ops")
      return legacy_ops;
    if (domain == "api")
      return fs::path(legacy_ops.string() + ".api.csv");
    if (domain == "kernels")
      return fs::path(legacy_ops.string() + ".kernels.csv");
    if (domain == "vram")
      return fs::path(legacy_ops.string() + ".vram.csv");
    if (domain == "roctx")
      return fs::path(legacy_ops.string() + ".roctx.csv");
    if (domain == "summary")
      return fs::path(legacy_ops.string() + ".summary.txt");
    if (domain == "launch")
      return fs::path(legacy_ops.string() + ".launch.csv");
  }
  if (domain == "ops" || domain == "kernels")
    return fs::path(prefix.string() + "_kernel_trace.csv");
  if (domain == "api")
    return fs::path(prefix.string() + "_hip_api_trace.csv");
  if (domain == "roctx")
    return fs::path(prefix.string() + "_roctx_trace.csv");
  if (domain == "copy")
    return fs::path(prefix.string() + "_memory_copy_trace.csv");
  if (domain == "alloc")
    return fs::path(prefix.string() + "_memory_allocation_trace.csv");
  if (domain == "agents")
    return fs::path(prefix.string() + "_agent_info.csv");
  if (domain == "launch")
    return fs::path(prefix.string() + "_ktrace_launch_info.csv");
  if (domain == "runtime")
    return fs::path(prefix.string() + "_hip_runtime_info.csv");
  if (domain == "cpu_sched")
    return fs::path(prefix.string() + "_cpu_sched.csv");
  if (domain == "cpu_samples")
    return fs::path(prefix.string() + "_cpu_samples.csv");
  if (domain == "cpu_images")
    return fs::path(prefix.string() + "_cpu_images.csv");
  if (domain == "results")
    return fs::path(prefix.string() + "_results.json");
  if (domain == "extensions")
    return fs::path(prefix.string() + "_ktrace_extensions.json");
  return {};
}

fs::path Capture::AutoOutput(const std::string &suffix) const {
  return fs::path(prefix.string() + suffix);
}

Capture ResolveCapture(const fs::path &source) {
  const std::string raw = source.string();
  for (const auto &suffix : kDomainSuffixes) {
    if (!EndsWithInsensitive(raw, suffix))
      continue;
    fs::path prefix(raw.substr(0, raw.size() - suffix.size()));
    if (fs::exists(source))
      return {prefix, source, {}};
  }
  if (EndsWithInsensitive(raw, "_ops.csv") && fs::exists(source))
    return {fs::path(raw.substr(0, raw.size() - 8)), source, source};
  for (const auto &suffix : kDomainSuffixes) {
    if (fs::exists(fs::path(raw + suffix)))
      return {source, source, {}};
  }
  if (fs::exists(fs::path(raw + "_ops.csv"))) {
    fs::path ops(raw + "_ops.csv");
    return {source, source, ops};
  }
  throw std::runtime_error(raw + ": capture prefix or domain file not found");
}

std::vector<Phase>
SegmentPhases(std::vector<std::pair<double, std::string>> marks,
              double gap_seconds) {
  std::sort(marks.begin(), marks.end());
  std::vector<Phase> out;
  std::string current;
  double start = 0, last = 0;
  for (const auto &[time, tag] : marks) {
    if (current.empty() || current != tag ||
        time - last > gap_seconds * 1'000'000.0) {
      if (!current.empty() && last - start > 50'000.0)
        out.push_back({current, start, last});
      current = tag;
      start = time;
    }
    last = time;
  }
  if (!current.empty() && last - start > 50'000.0)
    out.push_back({current, start, last});

  const double max_bridge = gap_seconds * 1'000'000.0;
  for (size_t i = 0; i + 1 < out.size(); ++i) {
    const double gap = out[i + 1].begin_us - out[i].end_us;
    if (gap < 0 || gap > max_bridge)
      continue;
    const double boundary = (out[i].end_us + out[i + 1].begin_us) * 0.5;
    out[i].end_us = boundary;
    out[i + 1].begin_us = boundary;
  }
  return out;
}

class ComgrDemangler {
public:
  ComgrDemangler() {
    char executable[MAX_PATH]{};
    if (GetModuleFileNameA(nullptr, executable, sizeof(executable))) {
      const fs::path bundled =
          fs::path(executable).parent_path() / "amd_comgr_3.dll";
      module_ = LoadLibraryW(bundled.c_str());
    }
    if (!module_) {
      const std::string explicit_path = Env("AMD_COMGR");
      if (!explicit_path.empty())
        module_ = LoadLibraryA(explicit_path.c_str());
    }
    if (!module_)
      module_ = LoadLibraryA("amd_comgr_3.dll");
    if (!module_)
      return;

    create_data_ = Load<CreateData>("amd_comgr_create_data");
    release_data_ = Load<ReleaseData>("amd_comgr_release_data");
    set_data_ = Load<SetData>("amd_comgr_set_data");
    get_data_ = Load<GetData>("amd_comgr_get_data");
    demangle_ = Load<Demangle>("amd_comgr_demangle_symbol_name");
    if (!create_data_ || !release_data_ || !set_data_ || !get_data_ ||
        !demangle_) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
  }

  ~ComgrDemangler() {
    if (module_)
      FreeLibrary(module_);
  }

  ComgrDemangler(const ComgrDemangler &) = delete;
  ComgrDemangler &operator=(const ComgrDemangler &) = delete;

  bool IsAvailable() const { return module_ != nullptr; }

  bool DemangleName(const std::string &mangled, std::string &demangled) const {
    if (!module_)
      return false;
    amd_comgr_data_t input{}, output{};
    if (create_data_(AMD_COMGR_DATA_KIND_BYTES, &input) !=
        AMD_COMGR_STATUS_SUCCESS)
      return false;

    bool success = false;
    if (set_data_(input, mangled.size(), mangled.data()) ==
            AMD_COMGR_STATUS_SUCCESS &&
        demangle_(input, &output) == AMD_COMGR_STATUS_SUCCESS) {
      size_t size = 0;
      if (get_data_(output, &size, nullptr) == AMD_COMGR_STATUS_SUCCESS &&
          size != 0) {
        std::string value(size, '\0');
        if (get_data_(output, &size, value.data()) ==
            AMD_COMGR_STATUS_SUCCESS) {
          value.resize(size);
          while (!value.empty() && value.back() == '\0')
            value.pop_back();
          demangled = std::move(value);
          success = !demangled.empty();
        }
      }
    }
    if (output.handle)
      release_data_(output);
    release_data_(input);
    return success;
  }

private:
  using CreateData = decltype(&amd_comgr_create_data);
  using ReleaseData = decltype(&amd_comgr_release_data);
  using SetData = decltype(&amd_comgr_set_data);
  using GetData = decltype(&amd_comgr_get_data);
  using Demangle = decltype(&amd_comgr_demangle_symbol_name);

  template <typename Function> Function Load(const char *name) {
    return reinterpret_cast<Function>(GetProcAddress(module_, name));
  }

  HMODULE module_ = nullptr;
  CreateData create_data_ = nullptr;
  ReleaseData release_data_ = nullptr;
  SetData set_data_ = nullptr;
  GetData get_data_ = nullptr;
  Demangle demangle_ = nullptr;
};

std::string FindCxxFilt() {
  char module[MAX_PATH]{};
  if (GetModuleFileNameA(nullptr, module, sizeof(module))) {
    const fs::path bundled =
        fs::path(module).parent_path() / "llvm-cxxfilt.exe";
    if (fs::exists(bundled))
      return bundled.string();
  }

  const std::string explicit_path = Env("CXXFILT");
  if (!explicit_path.empty() && fs::exists(explicit_path))
    return explicit_path;

  char found[MAX_PATH]{};
  _searchenv_s("llvm-cxxfilt.exe", "PATH", found, sizeof(found));
  if (found[0])
    return found;
  _searchenv_s("c++filt.exe", "PATH", found, sizeof(found));
  if (found[0])
    return found;

  for (const auto &[root, relatives] :
       std::vector<std::pair<std::string, std::vector<std::string>>>{
           {Env("LLVM_ROOT").empty() ? Env("LLVM_HOME") : Env("LLVM_ROOT"),
            {"bin/llvm-cxxfilt.exe"}},
           {Env("HIP"),
            {"lib/llvm/bin/llvm-cxxfilt.exe", "bin/llvm-cxxfilt.exe"}}}) {
    for (const auto &relative : relatives) {
      fs::path candidate = fs::path(root) / relative;
      if (!root.empty() && fs::exists(candidate))
        return candidate.string();
    }
  }
  return {};
}

std::unordered_map<std::string, std::string>
ResolveKernelNames(const std::vector<std::string> &raw_names) {
  std::vector<std::string> names;
  std::set<std::string> seen;
  for (const auto &name : raw_names)
    if (!name.empty() && seen.insert(name).second)
      names.push_back(name);

  std::unordered_map<std::string, std::string> result;
  bool demangled = false;
  ComgrDemangler comgr;
  if (comgr.IsAvailable() && !names.empty()) {
    demangled = true;
    for (const auto &name : names) {
      std::string value;
      if (!comgr.DemangleName(name, value)) {
        demangled = false;
        result.clear();
        break;
      }
      result[name] = KernelLabel(value);
    }
  }

  const std::string tool = demangled ? std::string{} : FindCxxFilt();
  if (!demangled && !tool.empty() && !names.empty()) {
    const std::string token = std::to_string(GetCurrentProcessId()) + "_" +
                              std::to_string(GetCurrentThreadId()) + "_" +
                              std::to_string(GetTickCount64());
    const fs::path input =
        fs::temp_directory_path() / ("ktrace_names_" + token + ".txt");
    const fs::path output =
        fs::temp_directory_path() / ("ktrace_names_" + token + ".out");
    {
      std::ofstream file(input, std::ios::binary);
      for (const auto &name : names)
        file << name << '\n';
    }
    // cmd.exe requires an extra outer quote when the executable path itself
    // is quoted; without it, "C:\Program Files\..." is truncated to C:\Program.
    const std::string command = "\"\"" + tool + "\" < \"" + input.string() +
                                "\" > \"" + output.string() + "\"\"";
    if (std::system(command.c_str()) == 0) {
      std::ifstream file(output, std::ios::binary);
      std::string line;
      size_t i = 0;
      while (i < names.size() && std::getline(file, line))
        result[names[i++]] = KernelLabel(Trim(line));
      demangled = result.size() == names.size();
    }
    std::error_code ec;
    fs::remove(input, ec);
    fs::remove(output, ec);
  }
  if (!demangled) {
    std::cerr << "[warn] amd_comgr and llvm-cxxfilt unavailable or failed; "
                 "kernel names only partly resolved\n";
    result.clear();
    for (const auto &name : names)
      result[name] = KernelFallback(name);
  }

  std::map<std::string, std::vector<std::string>> owners;
  for (const auto &[raw, label] : result)
    owners[label].push_back(raw);
  for (auto &[label, raws] : owners) {
    if (raws.size() < 2)
      continue;
    std::sort(raws.begin(), raws.end());
    std::set<std::string> common = SplitTokens(raws.front());
    for (size_t i = 1; i < raws.size(); ++i) {
      std::set<std::string> next, tokens = SplitTokens(raws[i]);
      std::set_intersection(common.begin(), common.end(), tokens.begin(),
                            tokens.end(), std::inserter(next, next.begin()));
      common.swap(next);
    }
    for (size_t i = 0; i < raws.size(); ++i) {
      std::vector<std::string> varying;
      for (const auto &token : SplitTokens(raws[i]))
        if (!common.count(token))
          varying.push_back(token);
      if (!varying.empty() && varying.size() <= 6) {
        std::ostringstream suffix;
        for (size_t j = 0; j < varying.size(); ++j)
          suffix << (j ? "," : "") << varying[j];
        result[raws[i]] = label + "[" + suffix.str() + "]";
      } else
        result[raws[i]] = label + "#" + std::to_string(i + 1);
    }
  }
  return result;
}
} // namespace ktrace
