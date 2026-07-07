/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mmap_file.hpp"

#ifdef _WIN32
#include "mmap_file_win.hpp"
#endif

namespace morphizen {
std::unique_ptr<MemFile>
MemFile::create([[maybe_unused]] const std::filesystem::path &path) {
#ifdef _WIN32
  return MemFileWin::create(path);
#else
  return nullptr;
#endif
}
MemFile::~MemFile() {}
} // namespace morphizen
