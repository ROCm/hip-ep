/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "./mmap_file_win.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <windows.h>
namespace vaip_core {
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
std::unique_ptr<MemFile> MemFileWin::create(const std::filesystem::path& path) {
  static_assert(sizeof(HANDLE) == sizeof(MemFileWin::handle_t), "64-bit only");
  auto handle = CreateFileW(
      path.wstring().c_str(),
      GENERIC_READ | GENERIC_WRITE, // for some reason doesn't work w/o write
      0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    std::string error_msg = "CreateFileW failed: " + GetLastErrorAsString();
    throw std::runtime_error(error_msg);
  }
  auto size = GetFileSize(
      handle, nullptr /* does not support file larger than 2G yet.*/);
  if (size == INVALID_FILE_SIZE) {
    CloseHandle(handle);
    std::string error_msg = "GetFileSize failed: " + GetLastErrorAsString();
    throw std::runtime_error(error_msg);
  }
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
                       void* base)
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
void* MemFileWin::base() { return m_base; }
size_t MemFileWin::size() const { return m_size; }
} // namespace vaip_core
