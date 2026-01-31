/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <cstdio>
#include <io.h>
#include <string>
#include <windows.h>
#include <winternl.h>
#include <fileapi.h>
// clang-format on
//
#include "glog/logging.h"
#include "morphizen/env_config.hpp"
#include "morphizen/util.hpp"

DEF_ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE, "1")
DEF_ENV_PARAM(MORPHIZEN_DEBUG_POSIX_DELETE, "0")

namespace morphizen {

// NT API structures and constants for POSIX delete
#ifndef FILE_DISPOSITION_DELETE
#  define FILE_DISPOSITION_DELETE 0x00000001
#endif
#ifndef FILE_DISPOSITION_POSIX_SEMANTICS
#  define FILE_DISPOSITION_POSIX_SEMANTICS 0x00000002
#endif
#ifndef FileDispositionInformationEx
#  define FileDispositionInformationEx 64
#endif

typedef struct _FILE_DISPOSITION_INFORMATION_EX {
  ULONG Flags;
} FILE_DISPOSITION_INFORMATION_EX, *PFILE_DISPOSITION_INFORMATION_EX;

typedef struct _IO_STATUS_BLOCK {
  union {
    NTSTATUS Status;
    PVOID Pointer;
  };
  ULONG_PTR Information;
} IO_STATUS_BLOCK, *PIO_STATUS_BLOCK;

typedef NTSTATUS(WINAPI* NtSetInformationFileFunc)(
    HANDLE FileHandle, PIO_STATUS_BLOCK IoStatusBlock, PVOID FileInformation,
    ULONG Length, FILE_INFORMATION_CLASS FileInformationClass);

static std::string GetLastErrorAsString() {
  DWORD error_code = GetLastError();
  if (error_code == 0) {
    return std::string(); // No error occurred
  }

  LPVOID message_buffer;
  DWORD format_result = FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR)&message_buffer, 0, nullptr);

  if (format_result == 0) {
    return "Failed to format message for error code: " +
           std::to_string(error_code);
  }

  std::string message(static_cast<LPSTR>(message_buffer), format_result);

  LocalFree(message_buffer);

  return message;
}

static std::string GetFilePathFromHandle(HANDLE h) {
  if (h == INVALID_HANDLE_VALUE) {
    return "<invalid handle>";
  }

  // First call to get required buffer size
  DWORD required_size =
      GetFinalPathNameByHandleW(h, nullptr, 0, VOLUME_NAME_DOS);
  if (required_size == 0) {
    return "<failed to get path: " + GetLastErrorAsString() + ">";
  }

  // Allocate buffer and get the path
  std::vector<wchar_t> buffer(required_size + 1);
  DWORD result = GetFinalPathNameByHandleW(h, buffer.data(), required_size + 1,
                                           VOLUME_NAME_DOS);
  if (result == 0 || result > required_size) {
    return "<failed to get path: " + GetLastErrorAsString() + ">";
  }

  // Convert wide string to narrow string
  int size_needed = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, nullptr,
                                        0, nullptr, nullptr);
  if (size_needed == 0) {
    return "<failed to convert path to UTF-8>";
  }

  std::string path_utf8(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, path_utf8.data(),
                      size_needed, nullptr, nullptr);
  // Remove null terminator if present
  if (!path_utf8.empty() && path_utf8.back() == '\0') {
    path_utf8.pop_back();
  }

  return path_utf8;
}

MORPHIZEN_DLL_SPEC FILE* tmpfile_with_posix_delete() {
  // Create temporary file using tmpfile_s
  FILE* tmp_file = nullptr;
  errno_t err = tmpfile_s(&tmp_file);
  if (err != 0 || tmp_file == nullptr) {
    return tmp_file; // Return nullptr on failure
  }

  // Debug logging: tmpfile created
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_POSIX_DELETE) != 0)
      << "POSIX delete debug: tmpfile created, FILE*=" << tmp_file;

  // Check if POSIX delete is enabled
  if (ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) == 0) {
    return tmp_file; // POSIX delete disabled, return standard tmpfile
  }

  // Extract Windows HANDLE from FILE*
  int fd = _fileno(tmp_file);
  if (fd == -1) {
    LOG(WARNING) << "Failed to get file descriptor from tmpfile: "
                 << GetLastErrorAsString();
    return tmp_file; // Continue with standard behavior
  }

  HANDLE temp_file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (temp_file_handle == INVALID_HANDLE_VALUE) {
    LOG(WARNING) << "Failed to get Windows handle from file descriptor: "
                 << GetLastErrorAsString();
    return tmp_file; // Continue with standard behavior
  }

  // Debug logging: original file path (before POSIX delete)
  std::string original_path;
  if (ENV_PARAM(MORPHIZEN_DEBUG_POSIX_DELETE) != 0) {
    original_path = GetFilePathFromHandle(temp_file_handle);
    LOG(INFO) << "POSIX delete debug: original path: " << original_path
              << ", handle=0x" << std::hex << temp_file_handle << std::dec;
  }

  // Attempt POSIX delete - gracefully handle failures
  HANDLE reopened_handle = INVALID_HANDLE_VALUE;
  try {
    // ReOpenFile creates a separate file object for the same file
    reopened_handle =
        ReOpenFile(temp_file_handle, DELETE,
                   FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, 0);

    if (reopened_handle == INVALID_HANDLE_VALUE) {
      LOG_IF(WARNING, ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) != 0)
          << "ReOpenFile failed, POSIX delete not applied: "
          << GetLastErrorAsString();
      return tmp_file; // Continue with standard behavior
    }

    // Load NtSetInformationFile dynamically (available in ntdll.dll)
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll == nullptr) {
      CloseHandle(reopened_handle);
      LOG_IF(WARNING, ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) != 0)
          << "Failed to load ntdll.dll, POSIX delete not applied";
      return tmp_file;
    }

    NtSetInformationFileFunc NtSetInformationFile =
        reinterpret_cast<NtSetInformationFileFunc>(
            GetProcAddress(ntdll, "NtSetInformationFile"));
    if (NtSetInformationFile == nullptr) {
      CloseHandle(reopened_handle);
      LOG_IF(WARNING, ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) != 0)
          << "NtSetInformationFile not available, POSIX delete not applied "
             "(requires Windows 10 1809+)";
      return tmp_file;
    }

    // Set POSIX delete flags
    FILE_DISPOSITION_INFORMATION_EX disp{};
    disp.Flags = FILE_DISPOSITION_DELETE | FILE_DISPOSITION_POSIX_SEMANTICS;

    IO_STATUS_BLOCK io_status_block = {};
    NTSTATUS status = NtSetInformationFile(
        reopened_handle, &io_status_block, &disp, sizeof(disp),
        static_cast<FILE_INFORMATION_CLASS>(FileDispositionInformationEx));

    // Close the reopened handle to trigger deletion
    CloseHandle(reopened_handle);

    if (!NT_SUCCESS(status)) {
      LOG_IF(WARNING, ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) != 0)
          << "NtSetInformationFile failed (status: 0x" << std::hex << status
          << std::dec << "), POSIX delete not applied";
      return tmp_file; // Continue with standard behavior
    }

    // POSIX delete successful - file is now in $Extend\$Deleted
    // Original FILE* handle remains open and accessible

    // Debug logging: current file path (after POSIX delete)
    if (ENV_PARAM(MORPHIZEN_DEBUG_POSIX_DELETE) != 0) {
      std::string current_path = GetFilePathFromHandle(temp_file_handle);
      LOG(INFO) << "POSIX delete debug: POSIX delete applied successfully";
      LOG(INFO) << "POSIX delete debug: current path: " << current_path;
      if (!original_path.empty() && original_path != current_path) {
        LOG(INFO) << "POSIX delete debug: file moved from original location to "
                     "$Extend\\$Deleted";
      }
    }
  } catch (...) {
    if (reopened_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(reopened_handle);
    }
    LOG_IF(WARNING, ENV_PARAM(MORPHIZEN_ENABLE_POSIX_DELETE) != 0)
        << "Exception during POSIX delete, continuing with standard behavior";
    return tmp_file; // Continue with standard behavior
  }

  return tmp_file; // Return FILE* with POSIX delete applied
}

MORPHIZEN_DLL_SPEC std::filesystem::path get_morphizen_path() {
  wchar_t path[MAX_PATH];
  HMODULE hm = NULL;

  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                        (LPCSTR)&get_morphizen_path, &hm) == 0) {
    int ret = GetLastError();
    LOG(ERROR) << "GetModuleHandle failed, error = " << ret;
    return {};
  }
  if (GetModuleFileNameW(hm, path, sizeof(path) / sizeof(wchar_t)) == 0) {
    int ret = GetLastError();
    LOG(ERROR) << "GetModuleFileName failed, error = " << ret;
    return {};
  }
  return std::filesystem::path(path);
}

unsigned int get_tid() {
  return static_cast<unsigned int>(GetCurrentThreadId());
}

unsigned int get_pid() {
  return static_cast<unsigned int>(GetCurrentProcessId());
}
} // namespace morphizen
