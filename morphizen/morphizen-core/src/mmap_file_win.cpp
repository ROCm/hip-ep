/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mmap_file_win.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <windows.h>
namespace morphizen {
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
std::unique_ptr<MemFile> MemFileWin::create(const std::filesystem::path &path) {
  static_assert(sizeof(HANDLE) == sizeof(MemFileWin::handle_t), "64-bit only");
  auto handle = CreateFileW(
      path.wstring().c_str(),
      GENERIC_READ | GENERIC_WRITE, // for some reason doesn't work w/o write
      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    std::string error_msg = "CreateFileW \"" + path.u8string() +
                            "\" failed: " + GetLastErrorAsString();
    ;
    throw std::runtime_error(error_msg);
  }
  // Use GetFileSizeEx for correct 64-bit file sizes. GetFileSize with
  // lpFileSizeHigh=NULL returns only the low DWORD, so files >4 GB get a
  // truncated m_size while the mapping itself covers the full file. This
  // causes downstream offset checks against size() to reject valid regions.
  LARGE_INTEGER li_size;
  if (!GetFileSizeEx(handle, &li_size)) {
    std::string error_msg = "GetFileSizeEx failed: " + GetLastErrorAsString();
    CloseHandle(handle);
    throw std::runtime_error(error_msg);
  }
  auto size = static_cast<size_t>(li_size.QuadPart);
  auto map_handle = CreateFileMappingW(handle, nullptr, PAGE_READWRITE, 0,
                                       0, // map entire file
                                       nullptr);
  if (map_handle == nullptr) {
    CloseHandle(handle);
    std::string error_msg =
        "CreateFileMappingW failed: " + GetLastErrorAsString();
    throw std::runtime_error(error_msg);
  }
  auto base =
      MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS, 0, 0, 0 // map entire file
      );
  return std::make_unique<MemFileWin>(handle, map_handle, size, base);
}
MemFileWin::MemFileWin(handle_t handle, handle_t map_handle, size_t size,
                       void *base)
    : m_handle(handle),         //
      m_map_handle(map_handle), //
      m_size{size},             //
      m_base{base} {
  // do nothing
}
MemFileWin::~MemFileWin() {
  if (m_base != nullptr) {
    UnmapViewOfFile(m_base);
  }
  if (m_map_handle != nullptr) {
    CloseHandle(m_map_handle);
  }
  if (m_handle != nullptr) {
    CloseHandle(m_handle);
  }
}
void *MemFileWin::base() { return m_base; }
size_t MemFileWin::size() const { return m_size; }
} // namespace morphizen
