/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mmap_file.hpp"
#include <filesystem>
#include <memory>
namespace morphizen {
// create a memory map file
class MemFileWin : public MemFile {
  using handle_t = void *; // NOLINT
public:
  static std::unique_ptr<MemFile> create(const std::filesystem::path &path);

public:
  MemFileWin(handle_t handle, handle_t map_handle, size_t size, void *base);
  virtual ~MemFileWin();

private:
  virtual void *base() override final;
  virtual size_t size() const override final;

private:
  handle_t m_handle;
  handle_t m_map_handle;
  size_t m_size;
  void *m_base;
};
} // namespace morphizen
