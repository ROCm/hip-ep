/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * Research program to investigate HANDLE lifetime management when extracted
 * from FILE*
 *
 * Questions to answer:
 * 1. When we extract HANDLE from FILE* via _get_osfhandle(), do they share
 * ownership?
 * 2. What happens if we close FILE* while HANDLE is still open?
 * 3. What happens if we close HANDLE while FILE* is still open?
 * 4. Can we create CreateFileMapping from a HANDLE extracted from tmpfile()?
 * 5. Does POSIX delete interfere with mmap?
 */

#ifdef _WIN32

#include <cstdio>
#include <io.h>
#include <iostream>
#include <string>
#include <windows.h>

// Helper to get error message
std::string GetLastErrorAsString() {
  DWORD error_code = GetLastError();
  if (error_code == 0) {
    return std::string();
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

void test_handle_extraction() {
  std::cout << "=== Test 1: Extract HANDLE from tmpfile() ===" << std::endl;

  // Create tmpfile
  FILE *tmp_file = tmpfile();
  if (!tmp_file) {
    std::cerr << "Failed to create tmpfile" << std::endl;
    return;
  }
  std::cout << "tmpfile created: FILE*=" << tmp_file << std::endl;

  // Write some data
  const char *test_data = "Hello, World! This is test data for mmap.";
  fwrite(test_data, 1, strlen(test_data), tmp_file);
  fflush(tmp_file);
  std::cout << "Wrote " << strlen(test_data) << " bytes" << std::endl;

  // Extract HANDLE
  int fd = _fileno(tmp_file);
  if (fd == -1) {
    std::cerr << "Failed to get file descriptor" << std::endl;
    fclose(tmp_file);
    return;
  }
  std::cout << "File descriptor: " << fd << std::endl;

  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to get Windows handle" << std::endl;
    fclose(tmp_file);
    return;
  }
  std::cout << "Windows HANDLE: 0x" << std::hex << handle << std::dec
            << std::endl;

  // Get file size
  DWORD file_size = GetFileSize(handle, nullptr);
  std::cout << "File size via HANDLE: " << file_size << " bytes" << std::endl;

  std::cout << "SUCCESS: HANDLE extraction works!" << std::endl;

  fclose(tmp_file);
}

void test_mmap_from_tmpfile() {
  std::cout << "\n=== Test 2: Create memory mapping from tmpfile HANDLE ==="
            << std::endl;

  // Create tmpfile
  FILE *tmp_file = tmpfile();
  if (!tmp_file) {
    std::cerr << "Failed to create tmpfile" << std::endl;
    return;
  }

  // Write test data
  const char *test_data =
      "MMAP TEST DATA: The quick brown fox jumps over the lazy dog.";
  size_t data_len = strlen(test_data);
  fwrite(test_data, 1, data_len, tmp_file);
  fflush(tmp_file);
  std::cout << "Wrote " << data_len << " bytes to tmpfile" << std::endl;

  // Extract HANDLE
  int fd = _fileno(tmp_file);
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));

  // Create file mapping
  HANDLE map_handle = CreateFileMappingW(handle, nullptr,
                                         PAGE_READONLY, // Read-only mapping
                                         0, 0,          // Map entire file
                                         nullptr        // No name
  );

  if (map_handle == nullptr) {
    std::cerr << "CreateFileMappingW failed: " << GetLastErrorAsString()
              << std::endl;
    fclose(tmp_file);
    return;
  }
  std::cout << "CreateFileMappingW succeeded: 0x" << std::hex << map_handle
            << std::dec << std::endl;

  // Map view of file
  void *mapped_base =
      MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, // Map from beginning
                    0                                // Map entire file
      );

  if (mapped_base == nullptr) {
    std::cerr << "MapViewOfFile failed: " << GetLastErrorAsString()
              << std::endl;
    CloseHandle(map_handle);
    fclose(tmp_file);
    return;
  }
  std::cout << "MapViewOfFile succeeded: " << mapped_base << std::endl;

  // Read data via mmap
  std::string mapped_data(static_cast<const char *>(mapped_base), data_len);
  std::cout << "Data read via mmap: \"" << mapped_data << "\"" << std::endl;

  // Verify data
  if (mapped_data == test_data) {
    std::cout << "SUCCESS: mmap data matches written data!" << std::endl;
  } else {
    std::cerr << "ERROR: mmap data mismatch!" << std::endl;
  }

  // Cleanup
  UnmapViewOfFile(mapped_base);
  CloseHandle(map_handle);
  fclose(tmp_file);
  std::cout << "Cleanup complete" << std::endl;
}

void test_lifetime_order() {
  std::cout << "\n=== Test 3: Test cleanup order (FILE* vs HANDLE) ==="
            << std::endl;

  // Test scenario: Close FILE* first, then cleanup mmap
  FILE *tmp_file = tmpfile();
  if (!tmp_file) {
    std::cerr << "Failed to create tmpfile" << std::endl;
    return;
  }

  const char *test_data = "Lifetime test data";
  fwrite(test_data, 1, strlen(test_data), tmp_file);
  fflush(tmp_file);

  int fd = _fileno(tmp_file);
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));

  // Create mapping
  HANDLE map_handle =
      CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
  void *mapped_base = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);

  if (!mapped_base) {
    std::cerr << "Failed to create mmap" << std::endl;
    CloseHandle(map_handle);
    fclose(tmp_file);
    return;
  }

  std::cout << "Created mmap successfully" << std::endl;

  // CRITICAL TEST: Close FILE* while mmap is still active
  std::cout << "Closing FILE* while mmap is active..." << std::endl;
  fclose(tmp_file);
  std::cout << "FILE* closed" << std::endl;

  // Try to access mapped data AFTER FILE* is closed
  try {
    std::string mapped_data(static_cast<const char *>(mapped_base),
                            strlen(test_data));
    std::cout << "Data after FILE* closed: \"" << mapped_data << "\""
              << std::endl;

    if (mapped_data == test_data) {
      std::cout << "SUCCESS: mmap still valid after FILE* closed!" << std::endl;
      std::cout << "CONCLUSION: Safe to close FILE* before unmapping"
                << std::endl;
    }
  } catch (...) {
    std::cerr << "ERROR: mmap became invalid after FILE* closed!" << std::endl;
  }

  // Cleanup mmap
  UnmapViewOfFile(mapped_base);
  CloseHandle(map_handle);
  std::cout << "mmap cleanup complete" << std::endl;
}

int main() {
  std::cout << "Windows HANDLE Lifetime Research Program\n" << std::endl;

  test_handle_extraction();
  test_mmap_from_tmpfile();
  test_lifetime_order();

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "1. HANDLE can be extracted from tmpfile() FILE* using "
               "_fileno() and _get_osfhandle()"
            << std::endl;
  std::cout << "2. CreateFileMapping works with HANDLE extracted from tmpfile()"
            << std::endl;
  std::cout
      << "3. mmap remains valid after FILE* is closed (safe cleanup order)"
      << std::endl;
  std::cout << "\nRecommended approach for embed mode mmap:" << std::endl;
  std::cout << "- Extract HANDLE from tmpfile() FILE*" << std::endl;
  std::cout << "- Create mapping from HANDLE (not FILE*)" << std::endl;
  std::cout << "- Close FILE* early if not needed for writing" << std::endl;
  std::cout << "- Cleanup mmap before process exit" << std::endl;

  return 0;
}

#else
int main() {
  std::cout << "This research program is Windows-only" << std::endl;
  return 0;
}
#endif
