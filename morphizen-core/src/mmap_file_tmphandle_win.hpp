/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mmap_file.hpp"
#include <cstdio>
#include <memory>

namespace morphizen {

/**
 * @brief Memory-mapped file wrapper for tmpfile() FILE* handles on Windows
 *
 * This class enables memory mapping of temporary files created by tmpfile().
 * It extracts the Windows HANDLE from a FILE* pointer and creates a memory
 * mapping using CreateFileMapping/MapViewOfFile.
 *
 * Lifetime management:
 * - The original FILE* can be closed before this object is destroyed
 * - The memory mapping remains valid after FILE* is closed
 * - This class manages its own HANDLE reference
 *
 * Usage:
 *   FILE* tmp = tmpfile();
 *   fwrite(data, size, 1, tmp);
 *   fflush(tmp);
 *   auto mem_file = MemFileTmpHandle::create(tmp);
 *   fclose(tmp);  // Safe to close FILE* now
 *   // mem_file remains valid for reading via mmap
 */
class MemFileTmpHandle : public MemFile {
  using handle_t = void*; // Windows HANDLE

public:
  /**
   * @brief Creates a memory-mapped file from a tmpfile() FILE* pointer
   *
   * @param file FILE* from tmpfile() or tmpfile_s(), must be valid and flushed
   * @return unique_ptr to MemFileTmpHandle, or nullptr on failure
   *
   * @note The FILE* can be safely closed after this function returns
   * @note This is Windows-only functionality (returns nullptr on other
   * platforms)
   */
  static std::unique_ptr<MemFile> create(FILE* file);

public:
  MemFileTmpHandle(handle_t file_handle, handle_t map_handle, size_t size,
                   void* base);
  virtual ~MemFileTmpHandle();

private:
  virtual void* base() override final;
  virtual size_t size() const override final;

private:
  handle_t
      m_file_handle;     // Extracted from FILE*, NOT owned (just for debugging)
  handle_t m_map_handle; // Created by CreateFileMapping, owned
  size_t m_size;
  void* m_base;
};

} // namespace morphizen
