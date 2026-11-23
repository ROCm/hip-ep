/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <tlhelp32.h>
#include <winver.h>

#include "ktrace_ipc.h"
#include "rdp_capture_abi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace rdp_abi;

constexpr DWORD kTimeoutExitCode = 124;

enum class AutoCaptureMode {
  none,
  frame,
  dispatch,
};

struct Options {
  AutoCaptureMode auto_mode = AutoCaptureMode::none;
  std::uint32_t auto_start = 1;
  std::uint32_t auto_count = 1;
  std::uint32_t auto_delay_ms = 0;
  std::uint32_t timeout_ms = 1800000;
  std::int32_t sqtt_buffer_size = 0;
  bool instruction_tracing = false;
  bool counter_collection = false;
  bool shader_instrumentation = false;
  bool rdp_only = false;
  bool etw = false;
  bool verbose = false;
  bool show_help = false;
  bool show_version = false;
  fs::path output_path;
  fs::path working_directory;
  std::vector<std::wstring> application_args;
};

struct ChildProcess {
  HANDLE process = nullptr;
  HANDLE job = nullptr;
  DWORD process_id = 0;
  std::uint64_t resume_timestamp_ns = 0;

  ~ChildProcess() {
    if (process != nullptr) {
      CloseHandle(process);
    }
    if (job != nullptr) {
      CloseHandle(job);
    }
  }

  ChildProcess() = default;
  ChildProcess(const ChildProcess &) = delete;
  ChildProcess &operator=(const ChildProcess &) = delete;
};

struct CollectorProcess {
  PROCESS_INFORMATION process{};
  ktrace::ipc::Mapping mapping;
  std::wstring session_name;

  ~CollectorProcess() {
    if (process.hThread)
      CloseHandle(process.hThread);
    if (process.hProcess)
      CloseHandle(process.hProcess);
    ktrace::ipc::Close(&mapping);
  }

  CollectorProcess() = default;
  CollectorProcess(const CollectorProcess &) = delete;
  CollectorProcess &operator=(const CollectorProcess &) = delete;
};

std::wstring WindowsError(DWORD error) {
  wchar_t *buffer = nullptr;
  const DWORD length = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error, 0, reinterpret_cast<wchar_t *>(&buffer), 0, nullptr);
  std::wstring message = (length != 0 && buffer != nullptr)
                             ? std::wstring(buffer, length)
                             : L"unknown error";
  if (buffer != nullptr) {
    LocalFree(buffer);
  }
  while (!message.empty() &&
         (message.back() == L'\r' || message.back() == L'\n')) {
    message.pop_back();
  }
  return message;
}

void PrintUsage() {
  std::wcout
      << L"ktrace - HIP runtime and kernel tracer with optional RGP capture\n"
      << L"Usage:\n"
      << L"  ktrace [options] -- <application> [arguments]\n\n"
      << L"RGP options (RGP is enabled only when -a is present):\n"
      << L"  -a, --rgp-auto-capture <value>       frame[:N] or "
         L"dispatch[:start[:count]]\n"
      << L"      --rgp-auto-capture-delay <ms>    Delay before dispatch "
         L"auto-capture\n"
      << L"      --rgp-instruction-tracing        Enable instruction tracing\n"
      << L"      --rgp-counter-collection         Enable SPM counter "
         L"collection\n"
      << L"      --rgp-shader-instrumentation     Enable shader "
         L"instrumentation\n"
      << L"      --rgp-sqtt-buffer-size <value>   "
         L"minimum/low/default/high/maximum\n\n"
      << L"Launch and output options:\n"
      << L"  -o, --output <file>                  Output .rgp file (requires "
         L"-a)\n"
      << L"      --working-directory <dir>        Child process working "
         L"directory\n"
      << L"      --timeout-minutes <min>          Overall timeout (default: "
         L"30)\n"
      << L"      --rdp-only                       Skip ktrace tracer "
         L"injection\n"
      << L"      --etw                            Collect CPU "
         L"scheduling/profile via ETW\n"
      << L"      --verbose                        Print verbose API logs\n"
      << L"  -m, --mode profiling                Accepted for official-CLI "
         L"compatibility\n"
      << L"  -h, --help                           Show this help\n"
      << L"  -v, --version                        Show version information\n\n"
      << L"Without -a, only ktrace is enabled and RdpCaptureApi.dll is not "
         L"loaded.\n"
      << L"When -o is omitted, the RGP file is written to KTRACE_OUTPUT_DIR "
         L"when set,\n"
      << L"or under the project output directory otherwise.\n"
      << L"ktrace_tracer.dll and optional RdpCaptureApi.dll are loaded beside "
         L"ktrace.exe.\n";
}

bool ParseUnsigned(const std::wstring &text, std::uint32_t *value) {
  if (text.empty()) {
    return false;
  }
  wchar_t *end = nullptr;
  errno = 0;
  const unsigned long long parsed = _wcstoui64(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end != L'\0' ||
      parsed > (std::numeric_limits<std::uint32_t>::max)()) {
    return false;
  }
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

std::vector<std::wstring> Split(const std::wstring &text, wchar_t delimiter) {
  std::vector<std::wstring> parts;
  std::size_t begin = 0;
  for (;;) {
    const std::size_t end = text.find(delimiter, begin);
    parts.emplace_back(text.substr(begin, end - begin));
    if (end == std::wstring::npos) {
      break;
    }
    begin = end + 1;
  }
  return parts;
}

bool ParseAutoCapture(const std::wstring &text, Options *options,
                      std::wstring *error) {
  const std::vector<std::wstring> parts = Split(text, L':');
  if (parts.empty()) {
    *error = L"empty auto-capture specification";
    return false;
  }

  if (parts[0] == L"frame") {
    if (parts.size() > 2) {
      *error = L"frame capture must be frame[:N]";
      return false;
    }
    std::uint32_t frame = 0;
    if (parts.size() == 2 && !ParseUnsigned(parts[1], &frame)) {
      *error = L"frame index must be a non-negative integer";
      return false;
    }
    options->auto_mode = AutoCaptureMode::frame;
    options->auto_start = frame;
    options->auto_count = 1;
    return true;
  }

  if (parts[0] == L"dispatch") {
    if (parts.size() > 3) {
      *error = L"dispatch capture must be dispatch[:start[:count]]";
      return false;
    }
    std::uint32_t start = 1;
    std::uint32_t count = 1;
    if (parts.size() >= 2 && (!ParseUnsigned(parts[1], &start) || start == 0)) {
      *error = L"dispatch start must be an integer greater than zero";
      return false;
    }
    if (parts.size() == 3 && (!ParseUnsigned(parts[2], &count) || count == 0)) {
      *error = L"dispatch count must be an integer greater than zero";
      return false;
    }
    options->auto_mode = AutoCaptureMode::dispatch;
    options->auto_start = start;
    options->auto_count = count;
    return true;
  }

  *error = L"auto-capture mode must be frame or dispatch";
  return false;
}

bool ParseSqttBufferSize(const std::wstring &text, std::int32_t *value) {
  if (text == L"default") {
    *value = 0;
  } else if (text == L"minimum" || text == L"min") {
    *value = 1;
  } else if (text == L"low") {
    *value = 2;
  } else if (text == L"high") {
    *value = 3;
  } else if (text == L"maximum" || text == L"max") {
    *value = 4;
  } else {
    return false;
  }
  return true;
}

bool InlineValue(const std::wstring &argument, const wchar_t *option,
                 std::wstring *value) {
  const std::wstring prefix = std::wstring(option) + L"=";
  if (argument.rfind(prefix, 0) != 0) {
    return false;
  }
  *value = argument.substr(prefix.size());
  return true;
}

bool ReadOptionValue(int argc, wchar_t **argv, int *index,
                     const std::wstring &argument, const wchar_t *long_name,
                     const wchar_t *short_name, std::wstring *value,
                     std::wstring *error) {
  if (InlineValue(argument, long_name, value)) {
    if (value->empty()) {
      *error = std::wstring(long_name) + L" requires a value";
      return false;
    }
    return true;
  }
  if (argument != long_name &&
      (short_name == nullptr || argument != short_name)) {
    return false;
  }
  if (*index + 1 >= argc) {
    *error = std::wstring(long_name) + L" requires a value";
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

bool ParseCommandLine(int argc, wchar_t **argv, Options *options,
                      std::wstring *error) {
  int application_index = -1;
  for (int i = 1; i < argc; ++i) {
    const std::wstring argument = argv[i];
    if (argument == L"--") {
      application_index = i + 1;
      break;
    }
    if (argument == L"-h" || argument == L"--help" ||
        argument.rfind(L"--help=", 0) == 0) {
      options->show_help = true;
      continue;
    }
    if (argument == L"-v" || argument == L"--version") {
      options->show_version = true;
      continue;
    }
    if (argument == L"--verbose") {
      options->verbose = true;
      continue;
    }
    if (argument == L"--rdp-only") {
      options->rdp_only = true;
      continue;
    }
    if (argument == L"--etw") {
      options->etw = true;
      continue;
    }
    if (argument == L"--rgp-instruction-tracing") {
      options->instruction_tracing = true;
      continue;
    }
    if (argument == L"--rgp-counter-collection") {
      options->counter_collection = true;
      continue;
    }
    if (argument == L"--rgp-shader-instrumentation") {
      options->shader_instrumentation = true;
      continue;
    }

    std::wstring value;
    std::wstring value_error;
    if (ReadOptionValue(argc, argv, &i, argument, L"--output", L"-o", &value,
                        &value_error)) {
      options->output_path = value;
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--timeout-minutes", nullptr,
                        &value, &value_error)) {
      std::uint32_t minutes = 0;
      constexpr std::uint32_t kMillisecondsPerMinute = 60000;
      if (!ParseUnsigned(value, &minutes) || minutes == 0 ||
          minutes > (std::numeric_limits<std::uint32_t>::max)() /
                        kMillisecondsPerMinute) {
        *error = L"--timeout-minutes must be an integer from 1 through 71582";
        return false;
      }
      options->timeout_ms = minutes * kMillisecondsPerMinute;
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    // Backward-compatible alias for scripts created before v0.3.
    if (ReadOptionValue(argc, argv, &i, argument, L"--rgp-auto-capture", L"-a",
                        &value, &value_error)) {
      if (!ParseAutoCapture(value, options, error)) {
        return false;
      }
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--rgp-auto-capture-delay",
                        nullptr, &value, &value_error)) {
      if (!ParseUnsigned(value, &options->auto_delay_ms)) {
        *error = L"--rgp-auto-capture-delay must be a non-negative integer";
        return false;
      }
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--rgp-sqtt-buffer-size",
                        nullptr, &value, &value_error)) {
      if (!ParseSqttBufferSize(value, &options->sqtt_buffer_size)) {
        *error = L"invalid --rgp-sqtt-buffer-size value";
        return false;
      }
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--timeout-ms", nullptr,
                        &value, &value_error)) {
      if (!ParseUnsigned(value, &options->timeout_ms) ||
          options->timeout_ms == 0) {
        *error = L"--timeout-ms must be an integer greater than zero";
        return false;
      }
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--working-directory",
                        nullptr, &value, &value_error)) {
      options->working_directory = value;
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }
    if (ReadOptionValue(argc, argv, &i, argument, L"--mode", L"-m", &value,
                        &value_error)) {
      if (value != L"profiling") {
        *error = L"only --mode profiling is supported";
        return false;
      }
      continue;
    }
    if (!value_error.empty()) {
      *error = value_error;
      return false;
    }

    *error = L"unknown option: " + argument;
    return false;
  }

  if (options->show_help || options->show_version) {
    return true;
  }
  const bool rdp_enabled = options->auto_mode != AutoCaptureMode::none;
  if (options->rdp_only && !rdp_enabled) {
    *error = L"--rdp-only requires --rgp-auto-capture";
    return false;
  }
  if (!rdp_enabled &&
      (options->instruction_tracing || options->counter_collection ||
       options->shader_instrumentation || options->auto_delay_ms != 0 ||
       options->sqtt_buffer_size != 0 || !options->output_path.empty())) {
    *error = L"RGP options require --rgp-auto-capture";
    return false;
  }
  if (application_index < 0 || application_index >= argc) {
    *error = L"an application must follow --";
    return false;
  }
  for (int i = application_index; i < argc; ++i) {
    options->application_args.emplace_back(argv[i]);
  }
  return true;
}

fs::path ExecutableDirectory() {
  std::vector<wchar_t> buffer(32768);
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                          static_cast<DWORD>(buffer.size()));
  if (length == 0 || length >= buffer.size()) {
    return {};
  }
  return fs::path(std::wstring(buffer.data(), length)).parent_path();
}

std::wstring Timestamp() {
  SYSTEMTIME time{};
  GetLocalTime(&time);
  std::wostringstream stream;
  stream << std::setfill(L'0') << std::setw(4) << time.wYear << std::setw(2)
         << time.wMonth << std::setw(2) << time.wDay << L'_' << std::setw(2)
         << time.wHour << std::setw(2) << time.wMinute << std::setw(2)
         << time.wSecond << L'_' << std::setw(3) << time.wMilliseconds;
  return stream.str();
}

std::uint64_t QpcNowNs() {
  static const LARGE_INTEGER frequency = [] {
    LARGE_INTEGER value{};
    QueryPerformanceFrequency(&value);
    return value;
  }();
  LARGE_INTEGER counter{};
  QueryPerformanceCounter(&counter);
  return static_cast<std::uint64_t>(counter.QuadPart *
                                    (1'000'000'000.0 / frequency.QuadPart));
}

std::wstring EnvironmentValue(const wchar_t *name) {
  const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
  if (size == 0) {
    return {};
  }
  std::vector<wchar_t> buffer(size);
  const DWORD written = GetEnvironmentVariableW(name, buffer.data(), size);
  if (written == 0 || written >= size) {
    return {};
  }
  return std::wstring(buffer.data(), written);
}

bool StartCollector(const fs::path &executable, CollectorProcess *collector,
                    std::wstring *error) {
  collector->session_name = std::to_wstring(GetCurrentProcessId()) + L"_" +
                            std::to_wstring(QpcNowNs());
  if (!ktrace::ipc::Create(collector->session_name, &collector->mapping, error))
    return false;

  if (!SetEnvironmentVariableW(L"KTRACE_IPC_SESSION",
                               collector->session_name.c_str()) ||
      !SetEnvironmentVariableW(L"KTRACE_IPC_MODE", L"collector")) {
    *error = L"cannot configure collector environment: " +
             WindowsError(GetLastError());
    return false;
  }

  std::wstring command =
      L"\"" + executable.wstring() + L"\" --session " + collector->session_name;
  std::vector<wchar_t> mutable_command(command.begin(), command.end());
  mutable_command.push_back(L'\0');
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr,
                      nullptr, FALSE, CREATE_UNICODE_ENVIRONMENT, nullptr,
                      nullptr, &startup, &collector->process)) {
    *error =
        L"cannot start ktrace_collector.exe: " + WindowsError(GetLastError());
    return false;
  }

  const ULONGLONG deadline = GetTickCount64() + 5000;
  while (!collector->mapping.session->collector_ready.load(
      std::memory_order_acquire)) {
    if (WaitForSingleObject(collector->process.hProcess, 0) == WAIT_OBJECT_0) {
      *error = L"ktrace_collector.exe exited before becoming ready";
      return false;
    }
    if (GetTickCount64() >= deadline) {
      *error = L"timed out waiting for ktrace_collector.exe";
      return false;
    }
    Sleep(1);
  }
  return true;
}

bool StopCollector(CollectorProcess *collector, std::wstring *error) {
  if (!collector->process.hProcess)
    return true;
  collector->mapping.session->shutdown_requested.store(
      1, std::memory_order_release);
  const DWORD wait = WaitForSingleObject(collector->process.hProcess, 900000);
  if (wait != WAIT_OBJECT_0) {
    *error = L"collector did not replay its journal within 15 minutes";
    return false;
  }
  DWORD exit_code = 0;
  if (!GetExitCodeProcess(collector->process.hProcess, &exit_code) ||
      exit_code != 0) {
    *error = L"collector failed with exit code " + std::to_wstring(exit_code);
    return false;
  }
  return true;
}

bool WriteLaunchMetadata(const ChildProcess &child, std::wstring *error) {
  std::wstring output_dir = EnvironmentValue(L"KTRACE_OUTPUT_DIR");
  if (output_dir.empty())
    output_dir = L".";
  std::wstring prefix = EnvironmentValue(L"KTRACE_OUTPUT_FILE");
  if (prefix.empty())
    prefix = std::to_wstring(child.process_id);

  std::error_code directory_error;
  fs::create_directories(output_dir, directory_error);
  if (directory_error) {
    *error = L"cannot create launch metadata directory: " +
             fs::path(output_dir).wstring();
    return false;
  }

  const fs::path path =
      fs::path(output_dir) /
      (L"." + prefix + L"_" + std::to_wstring(child.process_id) +
       L"_ops.csv.launch.csv");
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    *error = L"cannot create launch metadata: " + path.wstring();
    return false;
  }
  file << "Process_Id,Resume_Timestamp_ns\n"
       << child.process_id << ',' << child.resume_timestamp_ns << '\n';
  return static_cast<bool>(file);
}

std::wstring QuoteCommandLineArgument(const std::wstring &argument) {
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return argument;
  }

  std::wstring quoted(1, L'"');
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(L'"');
      backslashes = 0;
      continue;
    }
    quoted.append(backslashes, L'\\');
    backslashes = 0;
    quoted.push_back(character);
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'"');
  return quoted;
}

std::wstring BuildCommandLine(const std::vector<std::wstring> &arguments) {
  std::wstring command_line;
  for (const std::wstring &argument : arguments) {
    if (!command_line.empty()) {
      command_line.push_back(L' ');
    }
    command_line += QuoteCommandLineArgument(argument);
  }
  return command_line;
}

std::uint64_t CreationTime(DWORD process_id) {
  const HANDLE process =
      OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
  if (process == nullptr) {
    return 0;
  }
  FILETIME created{}, exited{}, kernel{}, user{};
  const BOOL ok = GetProcessTimes(process, &created, &exited, &kernel, &user);
  CloseHandle(process);
  if (!ok) {
    return 0;
  }
  return (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) |
         created.dwLowDateTime;
}

std::set<DWORD> Descendants(DWORD root, std::uint64_t root_started) {
  const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return {};
  }

  std::vector<std::pair<DWORD, DWORD>> edges;
  PROCESSENTRY32W entry{};
  entry.dwSize = sizeof(entry);
  for (BOOL ok = Process32FirstW(snapshot, &entry); ok;
       ok = Process32NextW(snapshot, &entry)) {
    edges.emplace_back(entry.th32ProcessID, entry.th32ParentProcessID);
  }
  CloseHandle(snapshot);

  std::map<DWORD, std::uint64_t> started{{root, root_started}};
  std::map<DWORD, std::uint64_t> cache;
  auto born = [&cache](DWORD process_id) {
    const auto found = cache.find(process_id);
    if (found != cache.end()) {
      return found->second;
    }
    return cache.emplace(process_id, CreationTime(process_id)).first->second;
  };

  std::set<DWORD> result{root};
  for (bool grew = true; grew;) {
    grew = false;
    for (const auto &[process_id, parent_id] : edges) {
      if (result.count(parent_id) == 0 || result.count(process_id) != 0) {
        continue;
      }
      const std::uint64_t created = born(process_id);
      if (created == 0 || created < started[parent_id] ||
          created < root_started) {
        continue;
      }
      started[process_id] = created;
      result.insert(process_id);
      grew = true;
    }
  }
  result.erase(root);
  return result;
}

bool InjectTracer(HANDLE process, const fs::path &tracer_path) {
  const std::wstring path = tracer_path.wstring();
  const SIZE_T bytes = (path.size() + 1) * sizeof(wchar_t);
  void *remote = VirtualAllocEx(process, nullptr, bytes,
                                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (remote == nullptr) {
    std::wcerr << L"ERROR: VirtualAllocEx failed: "
               << WindowsError(GetLastError()) << L'\n';
    return false;
  }
  if (!WriteProcessMemory(process, remote, path.c_str(), bytes, nullptr)) {
    std::wcerr << L"ERROR: WriteProcessMemory failed: "
               << WindowsError(GetLastError()) << L'\n';
    VirtualFreeEx(process, remote, 0, MEM_RELEASE);
    return false;
  }

  const auto load_library = reinterpret_cast<LPTHREAD_START_ROUTINE>(
      GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"));
  const HANDLE thread =
      CreateRemoteThread(process, nullptr, 0, load_library, remote, 0, nullptr);
  if (thread == nullptr) {
    std::wcerr << L"ERROR: CreateRemoteThread failed: "
               << WindowsError(GetLastError()) << L'\n';
    VirtualFreeEx(process, remote, 0, MEM_RELEASE);
    return false;
  }

  WaitForSingleObject(thread, INFINITE);
  DWORD loaded = 0;
  GetExitCodeThread(thread, &loaded);
  CloseHandle(thread);
  VirtualFreeEx(process, remote, 0, MEM_RELEASE);
  if (loaded == 0) {
    std::wcerr << L"ERROR: LoadLibraryW failed in target for " << tracer_path
               << L'\n';
    return false;
  }
  return true;
}

bool ResolveApplication(const std::wstring &application, fs::path *resolved) {
  const bool has_path =
      application.find_first_of(L"\\/") != std::wstring::npos ||
      (application.size() >= 2 && application[1] == L':');
  std::vector<wchar_t> buffer(32768);
  if (has_path) {
    const DWORD length =
        GetFullPathNameW(application.c_str(), static_cast<DWORD>(buffer.size()),
                         buffer.data(), nullptr);
    if (length == 0 || length >= buffer.size()) {
      return false;
    }
    *resolved = std::wstring(buffer.data(), length);
  } else {
    const DWORD length =
        SearchPathW(nullptr, application.c_str(),
                    fs::path(application).has_extension() ? nullptr : L".exe",
                    static_cast<DWORD>(buffer.size()), buffer.data(), nullptr);
    if (length == 0 || length >= buffer.size()) {
      return false;
    }
    *resolved = std::wstring(buffer.data(), length);
  }
  return fs::is_regular_file(*resolved);
}

class CaptureSession {
public:
  CaptureSession(fs::path dll_path, fs::path output_path, bool verbose)
      : dll_path_(std::move(dll_path)), output_path_(std::move(output_path)),
        verbose_(verbose) {}

  ~CaptureSession() {
    Shutdown(false);
    if (capture_event_ != nullptr) {
      CloseHandle(capture_event_);
    }
  }

  CaptureSession(const CaptureSession &) = delete;
  CaptureSession &operator=(const CaptureSession &) = delete;

  bool Initialize(const Options &options) {
    capture_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (capture_event_ == nullptr) {
      std::wcerr << L"ERROR: CreateEventW failed: "
                 << WindowsError(GetLastError()) << L'\n';
      return false;
    }

    module_ = LoadLibraryExW(dll_path_.c_str(), nullptr,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                 LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module_ == nullptr) {
      std::wcerr << L"ERROR: failed to load " << dll_path_ << L": "
                 << WindowsError(GetLastError()) << L'\n';
      return false;
    }

    const auto get_fn_table = reinterpret_cast<GetFnTableFn>(
        GetProcAddress(module_, "RdpCaptureGetFnTable"));
    if (get_fn_table == nullptr) {
      std::wcerr << L"ERROR: RdpCaptureGetFnTable was not exported by "
                 << dll_path_ << L'\n';
      return false;
    }

    Result table_result = get_fn_table(2, 0, 0, &table_);
    if (table_result == kSuccess) {
      api_major_ = 2;
    } else {
      table_ = {};
      table_result = get_fn_table(1, 0, 0, &table_);
      if (table_result != kSuccess) {
        std::wcerr << L"ERROR: RdpCaptureGetFnTable failed for API 2.0.0 "
                      L"and 1.0.0: "
                   << table_result << L'\n';
        return false;
      }
      api_major_ = 1;
    }

    initialize_ = FunctionAt<InitializeFn>(table_, kInitializeSlot);
    enable_feature_ = FunctionAt<EnableFeatureFn>(table_, kEnableFeatureSlot);
    destroy_ = FunctionAt<DestroyFn>(table_, kDestroySlot);
    get_default_profiling_ =
        FunctionAt<GetDefaultProfilingParamsFn>(table_, kProfilingDefaultsSlot);
    set_profiling_ =
        FunctionAt<SetProfilingParamsFn>(table_, kProfilingSetParamsSlot);
    begin_trace_ = FunctionAt<BeginTraceFn>(table_, kProfilingBeginSlot);
    abort_trace_ = FunctionAt<AbortTraceFn>(table_, kProfilingAbortSlot);

    if (initialize_ == nullptr || enable_feature_ == nullptr ||
        destroy_ == nullptr || get_default_profiling_ == nullptr ||
        set_profiling_ == nullptr || begin_trace_ == nullptr ||
        abort_trace_ == nullptr) {
      std::wcerr << L"ERROR: API " << api_major_
                 << L".0.0 function table is incomplete\n";
      return false;
    }

    ContextInitParams init_params{};
    init_params.app_filter.filter = AppFilterCallback;
    init_params.app_filter.user_data = this;
    init_params.log_callback.log = LogCallback;
    init_params.log_callback.user_data = this;
    const Result initialize_result = initialize_(&init_params, &context_);
    if (initialize_result != kSuccess) {
      context_ = nullptr;
      std::wcerr << L"ERROR: RDP initialization failed: " << initialize_result
                 << L". Close Radeon Developer Panel/Service if it owns the "
                    L"context.\n";
      return false;
    }

    FeatureEnableParams enable_params{};
    enable_params.feature = kProfilingFeature;
    enable_params.trace_finished_callback.trace_finished =
        TraceFinishedCallback;
    enable_params.trace_finished_callback.user_data = this;
    if (options.auto_mode == AutoCaptureMode::frame) {
      enable_params.body.profiling.flags = kCaptureFrame;
      enable_params.body.profiling.frame_capture_index = options.auto_start;
    } else {
      enable_params.body.profiling.flags = kCaptureDispatch;
      enable_params.body.profiling.dispatch_start_index = options.auto_start;
      enable_params.body.profiling.dispatch_count = options.auto_count;
      enable_params.body.profiling.dispatch_capture_delay_ms =
          options.auto_delay_ms;
    }
    if (options.shader_instrumentation) {
      enable_params.body.profiling.flags |= kShaderInstrumentation;
    }

    const Result enable_result = enable_feature_(context_, &enable_params);
    if (enable_result != kSuccess) {
      std::wcerr << L"ERROR: enabling profiling failed: " << enable_result
                 << L'\n';
      return false;
    }

    ProfilingParams profiling_params{};
    get_default_profiling_(&profiling_params);
    profiling_params.sqtt_buffer_size = options.sqtt_buffer_size;
    if (options.instruction_tracing) {
      profiling_params.flags |= kInstructionTracing;
    }
    if (options.counter_collection) {
      profiling_params.flags |= kCounterCollection;
    }
    set_profiling_(context_, &profiling_params);

    std::wcout << L"RDP: using API " << api_major_ << L".0.0\n";
    initialized_ = true;
    return true;
  }

  void SetTargetProcessId(DWORD process_id) {
    target_started_.store(CreationTime(process_id), std::memory_order_release);
    target_process_id_.store(process_id, std::memory_order_release);
  }

  HANDLE CaptureEvent() const { return capture_event_; }

  bool CaptureSucceeded() const {
    return capture_result_.load(std::memory_order_acquire) == kSuccess &&
           file_written_.load(std::memory_order_acquire);
  }

  Result CaptureResult() const {
    return capture_result_.load(std::memory_order_acquire);
  }

  std::uint64_t CaptureSize() const {
    return capture_size_.load(std::memory_order_acquire);
  }

  void Abort() {
    if (context_ != nullptr && abort_trace_ != nullptr &&
        WaitForSingleObject(capture_event_, 0) != WAIT_OBJECT_0) {
      const Result result = abort_trace_(context_);
      if (verbose_) {
        std::wcerr << L"RDP: abort_trace returned " << result << L'\n';
      }
    }
  }

  void Shutdown(bool abort_pending) {
    if (context_ != nullptr) {
      if (abort_pending) {
        Abort();
      }
      destroy_(context_);
      context_ = nullptr;
    }
    initialized_ = false;
    if (module_ != nullptr) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
  }

private:
  static std::uint8_t __cdecl AppFilterCallback(void *user_data,
                                                const ProcessInfo *process_info,
                                                GpuApi) {
    const auto *self = static_cast<CaptureSession *>(user_data);
    const DWORD target =
        self->target_process_id_.load(std::memory_order_acquire);
    if (process_info == nullptr || target == 0) {
      return 0;
    }
    if (process_info->process_id == target) {
      return 1;
    }
    const std::uint64_t target_started =
        self->target_started_.load(std::memory_order_acquire);
    if (target_started == 0) {
      return 0;
    }
    return Descendants(target, target_started)
                       .count(process_info->process_id) != 0
               ? 1
               : 0;
  }

  static void __cdecl LogCallback(void *user_data, LogLevel level,
                                  const char *source, std::uint32_t process_id,
                                  const char *message) {
    auto *self = static_cast<CaptureSession *>(user_data);
    if (!self->verbose_ && level < 2) {
      return;
    }
    std::lock_guard<std::mutex> lock(self->log_mutex_);
    std::cerr << "RDP[" << level << "] "
              << (source != nullptr ? source : "unknown");
    if (process_id != 0) {
      std::cerr << " pid=" << process_id;
    }
    std::cerr << ": " << (message != nullptr ? message : "") << '\n';
  }

  static void __cdecl TraceFinishedCallback(void *user_data, Feature,
                                            ConnectionId, Result result,
                                            std::uint64_t size,
                                            const std::uint8_t *data) {
    auto *self = static_cast<CaptureSession *>(user_data);
    bool wrote_file = false;
    if (result == kSuccess && data != nullptr &&
        size <= static_cast<std::uint64_t>(
                    (std::numeric_limits<std::streamsize>::max)())) {
      std::ofstream output(self->output_path_,
                           std::ios::binary | std::ios::trunc);
      if (output) {
        output.write(reinterpret_cast<const char *>(data),
                     static_cast<std::streamsize>(size));
        output.flush();
        wrote_file = output.good();
      }
    }
    self->capture_size_.store(size, std::memory_order_release);
    self->file_written_.store(wrote_file, std::memory_order_release);
    self->capture_result_.store(result, std::memory_order_release);
    SetEvent(self->capture_event_);
  }

  fs::path dll_path_;
  fs::path output_path_;
  bool verbose_ = false;
  HMODULE module_ = nullptr;
  FunctionTable table_{};
  Context context_ = nullptr;
  InitializeFn initialize_ = nullptr;
  EnableFeatureFn enable_feature_ = nullptr;
  DestroyFn destroy_ = nullptr;
  GetDefaultProfilingParamsFn get_default_profiling_ = nullptr;
  SetProfilingParamsFn set_profiling_ = nullptr;
  BeginTraceFn begin_trace_ = nullptr;
  AbortTraceFn abort_trace_ = nullptr;
  HANDLE capture_event_ = nullptr;
  std::atomic<DWORD> target_process_id_{0};
  std::atomic<std::uint64_t> target_started_{0};
  std::atomic<Result> capture_result_{-1};
  std::atomic<std::uint64_t> capture_size_{0};
  std::atomic<bool> file_written_{false};
  std::uint32_t api_major_ = 0;
  bool initialized_ = false;
  std::mutex log_mutex_;
};

bool LaunchSuspended(const Options &options, const fs::path &tracer_path,
                     CaptureSession *session, ChildProcess *child,
                     std::wstring *error) {
  fs::path application;
  if (!ResolveApplication(options.application_args.front(), &application)) {
    *error = L"application was not found: " + options.application_args.front();
    return false;
  }

  std::wstring command_line = BuildCommandLine(options.application_args);
  std::vector<wchar_t> mutable_command_line(command_line.begin(),
                                            command_line.end());
  mutable_command_line.push_back(L'\0');

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process_info{};
  const wchar_t *working_directory = options.working_directory.empty()
                                         ? nullptr
                                         : options.working_directory.c_str();
  if (!CreateProcessW(application.c_str(), mutable_command_line.data(), nullptr,
                      nullptr, FALSE,
                      CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT, nullptr,
                      working_directory, &startup, &process_info)) {
    *error = L"CreateProcessW failed for " + application.wstring() + L": " +
             WindowsError(GetLastError());
    return false;
  }

  child->process = process_info.hProcess;
  child->process_id = process_info.dwProcessId;
  child->job = CreateJobObjectW(nullptr, nullptr);
  if (child->job &&
      !AssignProcessToJobObject(child->job, process_info.hProcess)) {
    CloseHandle(child->job);
    child->job = nullptr;
  }
  if (session != nullptr) {
    session->SetTargetProcessId(child->process_id);
  }
  if (!options.rdp_only) {
    std::wcout << L"[ktrace] pid " << child->process_id
               << L" suspended, injecting " << tracer_path << L'\n';
    if (!InjectTracer(process_info.hProcess, tracer_path)) {
      TerminateProcess(process_info.hProcess, 1);
      WaitForSingleObject(process_info.hProcess, 5000);
      CloseHandle(process_info.hThread);
      *error = L"failed to inject ktrace tracer";
      return false;
    }
  }
  child->resume_timestamp_ns = QpcNowNs();
  if (ResumeThread(process_info.hThread) == static_cast<DWORD>(-1)) {
    const DWORD resume_error = GetLastError();
    TerminateProcess(process_info.hProcess, resume_error);
    WaitForSingleObject(process_info.hProcess, 5000);
    CloseHandle(process_info.hThread);
    *error = L"ResumeThread failed: " + WindowsError(resume_error);
    return false;
  }
  CloseHandle(process_info.hThread);
  std::wcout << (options.rdp_only ? L"[rdp] target resumed without ktrace\n"
                                  : L"[ktrace] tracer armed, target resumed\n");
  return true;
}

bool FindHipLibraryPath(DWORD process_id, fs::path *path) {
  const HANDLE snapshot = CreateToolhelp32Snapshot(
      TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, process_id);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return false;
  }

  MODULEENTRY32W module{};
  module.dwSize = sizeof(module);
  bool found = false;
  if (Module32FirstW(snapshot, &module)) {
    do {
      std::wstring name = module.szModule;
      std::transform(name.begin(), name.end(), name.begin(), [](wchar_t value) {
        return static_cast<wchar_t>(std::towlower(value));
      });
      if (name.rfind(L"amdhip64", 0) == 0 && name.size() >= 4 &&
          name.compare(name.size() - 4, 4, L".dll") == 0) {
        *path = module.szExePath;
        found = true;
        break;
      }
    } while (Module32NextW(snapshot, &module));
  }
  CloseHandle(snapshot);
  return found;
}

std::wstring FileVersion(const fs::path &path) {
  DWORD ignored = 0;
  const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &ignored);
  if (!size)
    return L"unknown";
  std::vector<unsigned char> data(size);
  if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()))
    return L"unknown";
  VS_FIXEDFILEINFO *info = nullptr;
  UINT info_size = 0;
  if (!VerQueryValueW(data.data(), L"\\", reinterpret_cast<void **>(&info),
                      &info_size) ||
      !info || info_size < sizeof(VS_FIXEDFILEINFO) ||
      info->dwSignature != 0xFEEF04BD)
    return L"unknown";
  std::wostringstream version;
  version << HIWORD(info->dwFileVersionMS) << L'.'
          << LOWORD(info->dwFileVersionMS) << L'.'
          << HIWORD(info->dwFileVersionLS) << L'.'
          << LOWORD(info->dwFileVersionLS);
  return version.str();
}

std::string CsvField(std::string value) {
  size_t position = 0;
  while ((position = value.find('"', position)) != std::string::npos) {
    value.insert(position, 1, '"');
    position += 2;
  }
  return '"' + value + '"';
}

std::string Utf8(const std::wstring &value) {
  if (value.empty())
    return {};
  const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                       static_cast<int>(value.size()), nullptr,
                                       0, nullptr, nullptr);
  if (size <= 0)
    return {};
  std::string result(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), size, nullptr, nullptr);
  return result;
}

void WriteHipRuntimeInfo(DWORD process_id, const fs::path &runtime_path,
                         const std::wstring &version) {
  std::wstring output_dir = EnvironmentValue(L"KTRACE_OUTPUT_DIR");
  if (output_dir.empty())
    output_dir = L".";
  std::wstring prefix = EnvironmentValue(L"KTRACE_OUTPUT_FILE");
  if (prefix.empty())
    prefix = L"ktrace";
  const fs::path output =
      fs::path(output_dir) / (prefix + L"_hip_runtime_info.csv");
  std::error_code error;
  const bool write_header =
      !fs::exists(output, error) || fs::file_size(output, error) == 0;
  std::ofstream file(output, std::ios::binary | std::ios::app);
  if (!file)
    return;
  if (write_header)
    file << "\"Process_Id\",\"Hip_Runtime_Path\",\"File_Version\"\n";
  file << process_id << ',' << CsvField(runtime_path.u8string()) << ','
       << CsvField(Utf8(version)) << '\n';
}

void ReportHipLibrary(DWORD process_id, std::set<DWORD> *reported) {
  if (reported->count(process_id))
    return;
  fs::path path;
  if (!FindHipLibraryPath(process_id, &path))
    return;
  reported->insert(process_id);
  const std::wstring version = FileVersion(path);
  WriteHipRuntimeInfo(process_id, path, version);
  std::wcout << L"[ktrace] HIP PID " << process_id << L'\n'
             << L"  amdhip64 path    : " << path << L'\n'
             << L"  amdhip64 version : " << version << L'\n';
}

enum class WaitOutcome {
  completed,
  timed_out,
  failed,
};

bool JobHasActiveProcesses(HANDLE job) {
  if (!job)
    return false;
  JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info{};
  return QueryInformationJobObject(job, JobObjectBasicAccountingInformation,
                                   &info, sizeof(info), nullptr) &&
         info.ActiveProcesses != 0;
}

WaitOutcome WaitForCaptureAndProcess(const Options &options,
                                     const fs::path &tracer_path,
                                     CaptureSession *session,
                                     ChildProcess *child,
                                     DWORD *process_exit_code) {
  using Clock = std::chrono::steady_clock;
  const auto deadline =
      Clock::now() + std::chrono::milliseconds(options.timeout_ms);
  const std::uint64_t root_started = CreationTime(child->process_id);
  std::set<DWORD> injected_descendants;
  std::set<DWORD> reported_hip_processes;
  bool process_done = false;
  bool capture_done = session == nullptr;
  bool job_done = child->job == nullptr;

  while (!process_done || !capture_done || !job_done) {
    ReportHipLibrary(child->process_id, &reported_hip_processes);
    const std::set<DWORD> descendants =
        options.rdp_only ? std::set<DWORD>{}
                         : Descendants(child->process_id, root_started);
    for (const DWORD process_id : descendants) {
      ReportHipLibrary(process_id, &reported_hip_processes);
      if (injected_descendants.count(process_id)) {
        continue;
      }
      const HANDLE process = OpenProcess(
          PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION |
              PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
          FALSE, process_id);
      if (process == nullptr) {
        continue;
      }
      if (!InjectTracer(process, tracer_path)) {
        std::wcerr << L"WARNING: child pid " << process_id
                   << L" injection will be retried\n";
      } else {
        injected_descendants.insert(process_id);
        std::wcout << L"[ktrace] child pid " << process_id << L", injected\n";
      }
      CloseHandle(process);
    }
    if (child->job)
      job_done = !JobHasActiveProcesses(child->job);

    HANDLE handles[2]{};
    DWORD handle_count = 0;
    DWORD process_index = MAXDWORD;
    DWORD capture_index = MAXDWORD;
    if (!process_done) {
      process_index = handle_count;
      handles[handle_count++] = child->process;
    }
    if (!capture_done) {
      capture_index = handle_count;
      handles[handle_count++] = session->CaptureEvent();
    }
    if (handle_count == 0) {
      Sleep(20);
      continue;
    }

    const auto now = Clock::now();
    if (now >= deadline) {
      return WaitOutcome::timed_out;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const DWORD wait_ms = static_cast<DWORD>((std::min)(
        remaining.count(), static_cast<decltype(remaining.count())>(20)));
    const DWORD wait_result =
        WaitForMultipleObjects(handle_count, handles, FALSE, wait_ms);
    if (wait_result == WAIT_TIMEOUT) {
      continue;
    }
    if (wait_result < WAIT_OBJECT_0 ||
        wait_result >= WAIT_OBJECT_0 + handle_count) {
      std::wcerr << L"ERROR: WaitForMultipleObjects failed: "
                 << WindowsError(GetLastError()) << L'\n';
      return WaitOutcome::failed;
    }

    const DWORD signaled = wait_result - WAIT_OBJECT_0;
    if (signaled == process_index) {
      process_done = true;
      if (!GetExitCodeProcess(child->process, process_exit_code)) {
        std::wcerr << L"ERROR: GetExitCodeProcess failed: "
                   << WindowsError(GetLastError()) << L'\n';
        return WaitOutcome::failed;
      }
    }
    if (signaled == capture_index) {
      capture_done = true;
    }
  }
  return WaitOutcome::completed;
}

void StopOwnedProcess(ChildProcess *child) {
  if (child->job) {
    if (!TerminateJobObject(child->job, kTimeoutExitCode))
      std::wcerr << L"WARNING: failed to terminate target job: "
                 << WindowsError(GetLastError()) << L'\n';
    WaitForSingleObject(child->process, 5000);
    return;
  }
  if (child->process == nullptr ||
      WaitForSingleObject(child->process, 0) == WAIT_OBJECT_0) {
    return;
  }
  if (!TerminateProcess(child->process, kTimeoutExitCode)) {
    std::wcerr << L"WARNING: failed to terminate launched process "
               << child->process_id << L": " << WindowsError(GetLastError())
               << L'\n';
    return;
  }
  WaitForSingleObject(child->process, 5000);
}
} // namespace

int wmain(int argc, wchar_t **argv) {
  Options options;
  std::wstring parse_error;
  if (!ParseCommandLine(argc, argv, &options, &parse_error)) {
    std::wcerr << L"ERROR: " << parse_error << L"\n\n";
    std::wcerr.flush();
    PrintUsage();
    return 2;
  }
  if (options.show_help) {
    PrintUsage();
    return 0;
  }
  if (options.show_version) {
    std::cout << "ktrace " KTRACE_VERSION_STRING
              << " (optional RdpCaptureApi ABI 2.0.0/1.0.0)\n";
    return 0;
  }

  const fs::path bin_directory = ExecutableDirectory();
  if (bin_directory.empty()) {
    std::wcerr << L"ERROR: could not determine ktrace.exe directory\n";
    return 3;
  }
  const fs::path project_directory = bin_directory.parent_path();
  const bool rdp_enabled = options.auto_mode != AutoCaptureMode::none;
  const fs::path dll_path = bin_directory / L"RdpCaptureApi.dll";
  const fs::path tracer_path = bin_directory / L"ktrace_tracer.dll";
  const fs::path collector_path = bin_directory / L"ktrace_collector.exe";
  const bool collector_enabled =
      !options.rdp_only &&
      _wcsicmp(EnvironmentValue(L"KTRACE_IPC_MODE").c_str(), L"inline") != 0;
  if (rdp_enabled && !fs::is_regular_file(dll_path)) {
    std::wcerr << L"ERROR: required DLL was not found: " << dll_path << L'\n';
    return 3;
  }
  if (!options.rdp_only && !fs::is_regular_file(tracer_path)) {
    std::wcerr << L"ERROR: required DLL was not found: " << tracer_path
               << L'\n';
    return 3;
  }
  if (collector_enabled && !fs::is_regular_file(collector_path)) {
    std::wcerr << L"ERROR: required collector was not found: " << collector_path
               << L'\n';
    return 3;
  }

  if (rdp_enabled && options.output_path.empty()) {
    const std::wstring ktrace_output_dir =
        EnvironmentValue(L"KTRACE_OUTPUT_DIR");
    if (!ktrace_output_dir.empty()) {
      std::wstring prefix = EnvironmentValue(L"KTRACE_OUTPUT_FILE");
      if (prefix.empty()) {
        prefix = L"ktrace";
      }
      options.output_path =
          fs::path(ktrace_output_dir) / (prefix + L"_capture.rgp");
    } else {
      options.output_path = project_directory / L"output" /
                            (L"profiling_" + Timestamp() + L".rgp");
    }
  } else if (rdp_enabled) {
    options.output_path = fs::absolute(options.output_path);
  }
  if (rdp_enabled && options.output_path.extension().empty()) {
    options.output_path += L".rgp";
  }
  if (rdp_enabled) {
    std::error_code directory_error;
    fs::create_directories(options.output_path.parent_path(), directory_error);
    if (directory_error) {
      std::wcerr << L"ERROR: could not create output directory "
                 << options.output_path.parent_path() << L": "
                 << directory_error.message().c_str() << L'\n';
      return 3;
    }
  }
  if (!options.working_directory.empty() &&
      !fs::is_directory(options.working_directory)) {
    std::wcerr << L"ERROR: working directory does not exist: "
               << options.working_directory << L'\n';
    return 2;
  }

  CollectorProcess collector;
  if (collector_enabled) {
    if (options.etw)
      SetEnvironmentVariableW(L"KTRACE_ETW", L"1");
    std::wstring collector_error;
    if (!StartCollector(collector_path, &collector, &collector_error)) {
      std::wcerr << L"ERROR: " << collector_error << L'\n';
      return 4;
    }
  }

  std::unique_ptr<CaptureSession> session;
  if (rdp_enabled) {
    session = std::make_unique<CaptureSession>(dll_path, options.output_path,
                                               options.verbose);
    if (!session->Initialize(options)) {
      std::wstring collector_error;
      StopCollector(&collector, &collector_error);
      return 4;
    }
  }

  ChildProcess child;
  std::wstring launch_error;
  if (!LaunchSuspended(options, tracer_path, session.get(), &child,
                       &launch_error)) {
    std::wcerr << L"ERROR: " << launch_error << L'\n';
    if (session) {
      session->Shutdown(true);
    }
    std::wstring collector_error;
    StopCollector(&collector, &collector_error);
    return 5;
  }
  if (collector_enabled) {
    collector.mapping.session->root_pid.store(child.process_id,
                                              std::memory_order_release);
    collector.mapping.session->resume_timestamp_ns.store(
        child.resume_timestamp_ns, std::memory_order_release);
  }
  if (!options.rdp_only) {
    std::wstring metadata_error;
    if (!WriteLaunchMetadata(child, &metadata_error)) {
      std::wcerr << L"WARNING: " << metadata_error << L'\n';
    }
  }

  if (rdp_enabled) {
    std::wcout << L"RDP: launched PID " << child.process_id << L'\n'
               << L"RDP: output " << options.output_path << L'\n';
  } else {
    std::wcout << L"[ktrace] launched PID " << child.process_id
               << L" without RGP capture\n";
  }
  std::wcout << std::flush;

  DWORD process_exit_code = STILL_ACTIVE;
  const WaitOutcome outcome = WaitForCaptureAndProcess(
      options, tracer_path, session.get(), &child, &process_exit_code);
  if (outcome != WaitOutcome::completed) {
    if (outcome == WaitOutcome::timed_out) {
      std::wcerr << L"ERROR: timed out after " << options.timeout_ms
                 << L" ms; aborting capture and stopping launched PID "
                 << child.process_id << L'\n';
    }
    if (session) {
      session->Abort();
      WaitForSingleObject(session->CaptureEvent(), 3000);
    }
    StopOwnedProcess(&child);
    std::wstring collector_error;
    if (!StopCollector(&collector, &collector_error))
      std::wcerr << L"WARNING: " << collector_error << L'\n';
    if (session) {
      session->Shutdown(false);
    }
    return outcome == WaitOutcome::timed_out ? kTimeoutExitCode : 6;
  }

  bool capture_failed = false;
  if (session) {
    const bool capture_succeeded = session->CaptureSucceeded();
    const Result capture_result = session->CaptureResult();
    const std::uint64_t capture_size = session->CaptureSize();
    session->Shutdown(false);

    if (capture_succeeded) {
      std::wcout << L"RDP: capture complete (" << capture_size << L" bytes)\n";
    } else {
      std::wcerr << L"ERROR: capture callback failed: result=" << capture_result
                 << L", bytes=" << capture_size << L'\n';
      capture_failed = true;
    }
  }
  if (collector_enabled) {
    std::wstring collector_error;
    if (!StopCollector(&collector, &collector_error)) {
      std::wcerr << L"ERROR: " << collector_error << L'\n';
      return capture_failed ? 7 : 9;
    }
  }
  if (capture_failed) {
    return 7;
  }
  if (process_exit_code != 0) {
    std::wcerr << L"WARNING: target exited with code " << process_exit_code
               << L'\n';
    return 8;
  }
  return 0;
}
