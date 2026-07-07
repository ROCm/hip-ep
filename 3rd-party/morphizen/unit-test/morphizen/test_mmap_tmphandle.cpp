/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../../morphizen-core/src/mmap_file_tmphandle_win.hpp"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

using namespace morphizen;

#ifdef _WIN32

TEST(MemFileTmpHandleTest, CreateFromTmpFile) {
  // Create tmpfile and write test data
  FILE *tmp = tmpfile();
  ASSERT_NE(tmp, nullptr) << "tmpfile() failed";

  const char *test_data = "Hello, mmap from tmpfile!";
  size_t data_len = strlen(test_data);

  size_t written = fwrite(test_data, 1, data_len, tmp);
  ASSERT_EQ(written, data_len) << "fwrite failed";
  fflush(tmp);

  // Create MemFileTmpHandle
  auto mem_file = MemFileTmpHandle::create(tmp);
  ASSERT_NE(mem_file, nullptr) << "MemFileTmpHandle::create failed";

  // Verify size
  EXPECT_EQ(mem_file->size(), data_len);

  // Verify base pointer is valid
  void *base = mem_file->base();
  ASSERT_NE(base, nullptr);

  // Verify data matches
  std::string mapped_data(static_cast<const char *>(base), data_len);
  EXPECT_EQ(mapped_data, test_data);

  // Close FILE* - mmap should remain valid
  fclose(tmp);

  // Verify data still accessible after FILE* closed
  std::string data_after_close(static_cast<const char *>(base), data_len);
  EXPECT_EQ(data_after_close, test_data);

  // mem_file destructor will cleanup the mapping
}

TEST(MemFileTmpHandleTest, EmptyFile) {
  // Create empty tmpfile
  FILE *tmp = tmpfile();
  ASSERT_NE(tmp, nullptr);
  fflush(tmp);

  // Should return nullptr for empty file
  auto mem_file = MemFileTmpHandle::create(tmp);
  EXPECT_EQ(mem_file, nullptr) << "Empty file should return nullptr";

  fclose(tmp);
}

TEST(MemFileTmpHandleTest, NullFilePointer) {
  // Passing null FILE* should return nullptr
  auto mem_file = MemFileTmpHandle::create(nullptr);
  EXPECT_EQ(mem_file, nullptr);
}

TEST(MemFileTmpHandleTest, LargeFile) {
  FILE *tmp = tmpfile();
  ASSERT_NE(tmp, nullptr);

  // Write 1MB of data
  const size_t size = 1024 * 1024;
  std::vector<char> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i % 256);
  }

  size_t written = fwrite(data.data(), 1, size, tmp);
  ASSERT_EQ(written, size);
  fflush(tmp);

  // Create mmap
  auto mem_file = MemFileTmpHandle::create(tmp);
  ASSERT_NE(mem_file, nullptr);
  EXPECT_EQ(mem_file->size(), size);

  // Verify data integrity
  const char *mapped = static_cast<const char *>(mem_file->base());
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(mapped[i], data[i]) << "Data mismatch at offset " << i;
    if (mapped[i] != data[i]) {
      break; // Stop after first mismatch to avoid flooding output
    }
  }

  fclose(tmp);
}

#else // Non-Windows

// MemFileTmpHandle is not implemented on non-Windows platforms.
// The source file is not compiled on Linux, so we cannot test it.
// Windows-specific tests above verify the functionality where it exists.

#endif

// Note: APIExists test removed - platform-specific tests above are sufficient
// The API only exists on Windows, so a cross-platform test would not compile on
// Linux
