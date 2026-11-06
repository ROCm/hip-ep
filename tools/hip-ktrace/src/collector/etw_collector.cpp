/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "etw_collector.h"

#include <evntcons.h>
#include <tdh.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace ktrace::collector {
namespace {
constexpr GUID kThreadGuid = {0x3d6fa8d1,
                              0xfe05,
                              0x11d0,
                              {0x9d, 0xda, 0x00, 0xc0, 0x4f, 0xd7, 0xba, 0x7c}};
constexpr GUID kImageLoadGuid = {
    0x2cb15d1d,
    0x5fc1,
    0x11d2,
    {0xab, 0xe1, 0x00, 0xa0, 0xc9, 0x11, 0xf5, 0x18}};
constexpr GUID kPerfInfoGuid = {
    0xce1dbfb4,
    0x137e,
    0x4da6,
    {0x87, 0xb0, 0x3f, 0x59, 0xaa, 0x10, 0x2c, 0xbc}};
constexpr GUID kStackWalkGuid = {
    0xdef2fe46,
    0x7bd6,
    0x4b80,
    {0xbd, 0x94, 0xf5, 0x7f, 0xe2, 0x0d, 0x0c, 0xe3}};
constexpr GUID kSystemTraceControlGuid = {
    0x9e814aad,
    0x3204,
    0x11d2,
    {0x9a, 0x82, 0x00, 0x60, 0x08, 0xa8, 0x69, 0x39}};
constexpr UCHAR kContextSwitchOpcode = 36;
constexpr UCHAR kSampledProfileOpcode = 46;

struct PropertiesBuffer {
  std::vector<unsigned char> bytes;
  EVENT_TRACE_PROPERTIES *properties = nullptr;
};

PropertiesBuffer MakeProperties(const std::wstring &session,
                                const std::wstring &file) {
  const std::size_t bytes =
      sizeof(EVENT_TRACE_PROPERTIES) +
      (session.size() + file.size() + 2) * sizeof(wchar_t);
  PropertiesBuffer result;
  result.bytes.resize(bytes);
  std::memset(result.bytes.data(), 0, result.bytes.size());
  result.properties =
      reinterpret_cast<EVENT_TRACE_PROPERTIES *>(result.bytes.data());
  result.properties->Wnode.BufferSize = static_cast<ULONG>(bytes);
  result.properties->Wnode.Guid = kSystemTraceControlGuid;
  result.properties->Wnode.ClientContext = 1; // QPC
  result.properties->Wnode.Flags = WNODE_FLAG_TRACED_GUID;
  result.properties->LogFileMode = EVENT_TRACE_FILE_MODE_SEQUENTIAL;
  result.properties->EnableFlags =
      EVENT_TRACE_FLAG_PROCESS | EVENT_TRACE_FLAG_THREAD |
      EVENT_TRACE_FLAG_IMAGE_LOAD | EVENT_TRACE_FLAG_CSWITCH |
      EVENT_TRACE_FLAG_PROFILE;
  result.properties->LoggerNameOffset = sizeof(EVENT_TRACE_PROPERTIES);
  result.properties->LogFileNameOffset =
      result.properties->LoggerNameOffset +
      static_cast<ULONG>((session.size() + 1) * sizeof(wchar_t));
  auto *logger = reinterpret_cast<wchar_t *>(
      result.bytes.data() + result.properties->LoggerNameOffset);
  auto *logfile = reinterpret_cast<wchar_t *>(
      result.bytes.data() + result.properties->LogFileNameOffset);
  wcscpy_s(logger, session.size() + 1, session.c_str());
  wcscpy_s(logfile, file.size() + 1, file.c_str());
  return result;
}

std::string Win32Error(ULONG code) {
  char *text = nullptr;
  const DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<char *>(&text), 0, nullptr);
  std::string result =
      size && text ? std::string(text, size) : "error " + std::to_string(code);
  if (text)
    LocalFree(text);
  while (!result.empty() && (result.back() == '\r' || result.back() == '\n'))
    result.pop_back();
  return result;
}

void EnableSystemProfilePrivilege() {
  HANDLE token = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(),
                        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
    return;
  LUID luid{};
  if (LookupPrivilegeValueW(nullptr, L"SeSystemProfilePrivilege", &luid)) {
    TOKEN_PRIVILEGES privileges{};
    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Luid = luid;
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr);
  }
  CloseHandle(token);
}

template <typename T>
bool Property(PEVENT_RECORD event, const wchar_t *name, T *value) {
  PROPERTY_DATA_DESCRIPTOR descriptor{};
  descriptor.PropertyName = reinterpret_cast<ULONGLONG>(name);
  descriptor.ArrayIndex = ULONG_MAX;
  ULONG size = sizeof(T);
  return TdhGetProperty(event, 0, nullptr, 1, &descriptor, size,
                        reinterpret_cast<PBYTE>(value)) == ERROR_SUCCESS;
}

struct Parser {
  std::ofstream sched;
  std::ofstream samples;
  std::ofstream images;
  LARGE_INTEGER frequency{};
  const std::set<std::uint32_t> *process_ids = nullptr;
  const std::set<std::uint32_t> *thread_ids = nullptr;

  static void WINAPI OnEvent(PEVENT_RECORD event) {
    static_cast<Parser *>(event->UserContext)->Event(event);
  }

  std::uint64_t Ns(LONGLONG ticks) const {
    return static_cast<std::uint64_t>(ticks *
                                      (1'000'000'000.0 / frequency.QuadPart));
  }

  void Event(PEVENT_RECORD event) {
    const UCHAR opcode = event->EventHeader.EventDescriptor.Opcode;
    if (IsEqualGUID(event->EventHeader.ProviderId, kThreadGuid) &&
        opcode == kContextSwitchOpcode) {
      ULONG old_thread = 0, new_thread = 0;
      UCHAR old_state = 0, wait_reason = 0;
      if (event->UserDataLength >= 16) {
        const auto *data = static_cast<const unsigned char *>(event->UserData);
        std::memcpy(&new_thread, data, 4);
        std::memcpy(&old_thread, data + 4, 4);
        wait_reason = data[12];
        old_state = data[14];
      } else {
        return;
      }
      if (!thread_ids->count(old_thread) && !thread_ids->count(new_thread))
        return;
      sched << Ns(event->EventHeader.TimeStamp.QuadPart) << ','
            << unsigned(event->BufferContext.ProcessorNumber) << ','
            << old_thread << ',' << new_thread << ',' << unsigned(old_state)
            << ',' << unsigned(wait_reason) << '\n';
      return;
    }

    if (IsEqualGUID(event->EventHeader.ProviderId, kPerfInfoGuid) &&
        opcode == kSampledProfileOpcode) {
      ULONG thread = 0;
      ULONGLONG ip = 0;
      if (event->UserDataLength >= 12) {
        const auto *data = static_cast<const unsigned char *>(event->UserData);
        if (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) {
          ULONG ip32 = 0;
          std::memcpy(&ip32, data, 4);
          ip = ip32;
          std::memcpy(&thread, data + 4, 4);
        } else {
          std::memcpy(&ip, data, 8);
          std::memcpy(&thread, data + 8, 4);
        }
      }
      if (!thread_ids->count(thread))
        return;
      samples << Ns(event->EventHeader.TimeStamp.QuadPart) << ','
              << unsigned(event->BufferContext.ProcessorNumber) << ','
              << event->EventHeader.ProcessId << ',' << thread << ",0x"
              << std::hex << ip << std::dec << ",\"0x" << std::hex << ip
              << std::dec << "\"\n";
      return;
    }

    if (IsEqualGUID(event->EventHeader.ProviderId, kStackWalkGuid) &&
        event->UserDataLength >= 16) {
      const auto *bytes = static_cast<const unsigned char *>(event->UserData);
      LONGLONG timestamp = 0;
      ULONG process = 0, thread = 0;
      std::memcpy(&timestamp, bytes, 8);
      std::memcpy(&process, bytes + 8, 4);
      std::memcpy(&thread, bytes + 12, 4);
      if (!thread_ids->count(thread))
        return;
      const bool is32 =
          (event->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0;
      const std::size_t width = is32 ? 4 : 8;
      samples << Ns(timestamp) << ','
              << unsigned(event->BufferContext.ProcessorNumber) << ','
              << process << ',' << thread << ",0x0,\"";
      for (std::size_t offset = 16; offset + width <= event->UserDataLength;
           offset += width) {
        ULONGLONG address = 0;
        std::memcpy(&address, bytes + offset, width);
        if (offset != 16)
          samples << ';';
        samples << "0x" << std::hex << address << std::dec;
      }
      samples << "\"\n";
      return;
    }

    if (IsEqualGUID(event->EventHeader.ProviderId, kImageLoadGuid)) {
      ULONG process = event->EventHeader.ProcessId;
      ULONGLONG base = 0, size = 0;
      std::wstring filename;
      const auto *data = static_cast<const unsigned char *>(event->UserData);
      const std::size_t name_offset =
          event->EventHeader.EventDescriptor.Version >= 3 ? 56 : 40;
      if (event->UserDataLength >= name_offset) {
        std::memcpy(&base, data, 8);
        std::memcpy(&size, data + 8, 8);
        std::memcpy(&process, data + 16, 4);
        const auto *name =
            reinterpret_cast<const wchar_t *>(data + name_offset);
        const std::size_t chars = (event->UserDataLength - name_offset) / 2;
        filename.assign(name, wcsnlen_s(name, chars));
      }
      if (!process_ids->count(process))
        return;
      images << process << ",0x" << std::hex << base << std::dec << ',' << size
             << ",\"";
      for (const wchar_t ch : filename)
        images << (ch < 0x80 ? static_cast<char>(ch) : '?');
      images << "\"\n";
    }
  }
};
} // namespace

EtwCollector::EtwCollector(fs::path output_dir, std::string prefix)
    : output_dir_(std::move(output_dir)), prefix_(std::move(prefix)) {
  session_name_ = L"NT Kernel Logger";
  etl_path_ = output_dir_ / (prefix_ + "_cpu.etl");
}

EtwCollector::~EtwCollector() {
  if (session_) {
    std::string ignored;
    static const std::set<std::uint32_t> empty;
    StopAndExport(empty, empty, &ignored);
  }
}

bool EtwCollector::Start(std::string *error) {
  fs::create_directories(output_dir_);
  EnableSystemProfilePrivilege();
  auto properties = MakeProperties(session_name_, etl_path_.wstring());
  const ULONG status =
      StartTraceW(&session_, session_name_.c_str(), properties.properties);
  if (status != ERROR_SUCCESS) {
    session_ = 0;
    *error = Win32Error(status);
    return false;
  }

  CLASSIC_EVENT_ID stack_event{};
  stack_event.EventGuid = kPerfInfoGuid;
  stack_event.Type = kSampledProfileOpcode;
  TraceSetInformation(session_, TraceStackTracingInfo, &stack_event,
                      sizeof(stack_event));
  return true;
}

bool EtwCollector::StopAndExport(const std::set<std::uint32_t> &process_ids,
                                 const std::set<std::uint32_t> &thread_ids,
                                 std::string *error) {
  if (!session_)
    return true;
  auto properties = MakeProperties(session_name_, etl_path_.wstring());
  const ULONG stop =
      ControlTraceW(session_, session_name_.c_str(), properties.properties,
                    EVENT_TRACE_CONTROL_STOP);
  session_ = 0;
  if (stop != ERROR_SUCCESS) {
    *error = Win32Error(stop);
    return false;
  }
  return Parse(process_ids, thread_ids, error);
}

bool EtwCollector::Parse(const std::set<std::uint32_t> &process_ids,
                         const std::set<std::uint32_t> &thread_ids,
                         std::string *error) {
  Parser parser;
  parser.process_ids = &process_ids;
  parser.thread_ids = &thread_ids;
  QueryPerformanceFrequency(&parser.frequency);
  parser.sched.open(output_dir_ / (prefix_ + "_cpu_sched.csv"),
                    std::ios::binary | std::ios::trunc);
  parser.samples.open(output_dir_ / (prefix_ + "_cpu_samples.csv"),
                      std::ios::binary | std::ios::trunc);
  parser.images.open(output_dir_ / (prefix_ + "_cpu_images.csv"),
                     std::ios::binary | std::ios::trunc);
  parser.sched
      << "Timestamp_ns,Cpu,Old_Thread,New_Thread,Old_State,Wait_Reason\n";
  parser.samples
      << "Timestamp_ns,Cpu,Process_Id,Thread_Id,Instruction_Pointer,Stack\n";
  parser.images << "Process_Id,Image_Base,Image_Size,File_Name\n";

  EVENT_TRACE_LOGFILEW logfile{};
  std::wstring path = etl_path_.wstring();
  logfile.LogFileName = path.data();
  logfile.ProcessTraceMode =
      PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
  logfile.EventRecordCallback = Parser::OnEvent;
  logfile.Context = &parser;
  TRACEHANDLE trace = OpenTraceW(&logfile);
  if (trace == INVALID_PROCESSTRACE_HANDLE) {
    *error = Win32Error(GetLastError());
    return false;
  }
  const ULONG status = ProcessTrace(&trace, 1, nullptr, nullptr);
  CloseTrace(trace);
  if (status != ERROR_SUCCESS) {
    *error = Win32Error(status);
    return false;
  }
  return true;
}
} // namespace ktrace::collector
