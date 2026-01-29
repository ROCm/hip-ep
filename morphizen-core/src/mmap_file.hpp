/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <memory>
namespace morphizen {
// create a memory map file
class MemFile {
public:
  static std::unique_ptr<MemFile> create(const std::filesystem::path& path);

public:
  virtual ~MemFile();
  virtual void* base() = 0;
  virtual size_t size() const = 0;
};
} // namespace morphizen
