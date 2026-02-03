/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mmap_file_tmphandle_win.hpp"

#ifdef _WIN32
#  include <glog/logging.h>
#  include <io.h>
#  include <memory>
#  include <string>
#  include <windows.h>

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

std::unique_ptr<MemFile> MemFileTmpHandle::create(FILE* file) {
  static_assert(sizeof(HANDLE) == sizeof(MemFileTmpHandle::handle_t),
                "64-bit only");

  if (!file) {
    LOG(WARNING) << "MemFileTmpHandle::create: FILE* is null";
    return nullptr;
  }

  // Extract file descriptor from FILE*
  int fd = _fileno(file);
  if (fd == -1) {
    LOG(WARNING) << "MemFileTmpHandle::create: _fileno failed";
    return nullptr;
  }

  // Extract Windows HANDLE from file descriptor
  HANDLE file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file_handle == INVALID_HANDLE_VALUE) {
    LOG(WARNING) << "MemFileTmpHandle::create: _get_osfhandle failed";
    return nullptr;
  }

  // Get file size
  DWORD size = GetFileSize(file_handle, nullptr);
  if (size == INVALID_FILE_SIZE) {
    std::string error_msg = "MemFileTmpHandle::create: GetFileSize failed: " +
                            GetLastErrorAsString();
    LOG(WARNING) << error_msg;
    return nullptr;
  }

  // Handle empty files
  if (size == 0) {
    LOG(INFO) << "MemFileTmpHandle::create: file is empty, cannot create "
                 "memory mapping";
    return nullptr;
  }

  // Create file mapping (read-only for tmpfile reading)
  HANDLE map_handle = CreateFileMappingW(file_handle, nullptr,
                                         PAGE_READONLY, // Read-only mapping
                                         0, 0,          // Map entire file
                                         nullptr);      // No name

  if (map_handle == nullptr) {
    std::string error_msg =
        "MemFileTmpHandle::create: CreateFileMappingW failed: " +
        GetLastErrorAsString();
    LOG(WARNING) << error_msg;
    return nullptr;
  }

  // Map view of file into memory
  void* base = MapViewOfFile(map_handle,
                             FILE_MAP_READ, // Read-only access
                             0, 0,          // Map from beginning
                             0);            // Map entire file

  if (base == nullptr) {
    std::string error_msg = "MemFileTmpHandle::create: MapViewOfFile failed: " +
                            GetLastErrorAsString();
    LOG(WARNING) << error_msg;
    CloseHandle(map_handle);
    return nullptr;
  }

  // Success: create MemFileTmpHandle object
  // Note: file_handle is NOT duplicated/owned - it's just stored for debugging
  // The mapping keeps the file data accessible even after FILE* is closed
  return std::make_unique<MemFileTmpHandle>(file_handle, map_handle, size,
                                            base);
}

MemFileTmpHandle::MemFileTmpHandle(handle_t file_handle, handle_t map_handle,
                                   size_t size, void* base)
    : m_file_handle(file_handle), // Not owned, just for reference
      m_map_handle(map_handle),   // Owned
      m_size{size},               //
      m_base{base} {
  // Nothing else to initialize
}

MemFileTmpHandle::~MemFileTmpHandle() {
  // Unmap view first
  if (m_base != nullptr) {
    UnmapViewOfFile(m_base);
  }

  // Close mapping handle (owned by this object)
  if (m_map_handle != nullptr) {
    CloseHandle(m_map_handle);
  }

  // Do NOT close m_file_handle - we don't own it
  // The original FILE* (if not already closed) or the OS will clean it up
}

void* MemFileTmpHandle::base() { return m_base; }

size_t MemFileTmpHandle::size() const { return m_size; }

} // namespace morphizen

#else // Non-Windows platforms

namespace morphizen {

std::unique_ptr<MemFile> MemFileTmpHandle::create(FILE* file) {
  // Not implemented on non-Windows platforms
  (void)file;
  return nullptr;
}

MemFileTmpHandle::MemFileTmpHandle(handle_t file_handle, handle_t map_handle,
                                   size_t size, void* base)
    : m_file_handle(file_handle), m_map_handle(map_handle), m_size{size},
      m_base{base} {}

MemFileTmpHandle::~MemFileTmpHandle() {}

void* MemFileTmpHandle::base() { return nullptr; }

size_t MemFileTmpHandle::size() const { return 0; }

} // namespace morphizen

#endif
