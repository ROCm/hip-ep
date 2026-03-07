/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen-foundation/file_io.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <io.h>
#endif

namespace hip {

// FileReader backed by a regular file on disk.
class DiskFileReader : public morphizen::FileReader {
public:
  explicit DiskFileReader(const char* path)
      : file_(std::fopen(path, "rb")), size_(0) {
    if (!file_)
      return;
#ifdef _WIN32
    // ftell returns long (32-bit on Windows), which overflows for files > 2GB.
    _fseeki64(file_, 0, SEEK_END);
    size_ = static_cast<std::size_t>(_ftelli64(file_));
    _fseeki64(file_, 0, SEEK_SET);
#else
    std::fseek(file_, 0, SEEK_END);
    size_ = static_cast<std::size_t>(std::ftell(file_));
    std::fseek(file_, 0, SEEK_SET);
#endif
  }

  ~DiskFileReader() override {
    if (file_)
      std::fclose(file_);
  }

  bool ok() const { return file_ != nullptr; }

  std::size_t size() const override { return size_; }

  void rewind() const override {
    if (file_)
      std::fseek(file_, 0, SEEK_SET);
  }

  std::size_t fread(void* buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fread(buffer, 1, size, file_);
  }

private:
  std::FILE* file_;
  std::size_t size_;
};

// FileWriter backed by a regular file on disk.
class DiskFileWriter : public morphizen::FileWriter {
public:
  explicit DiskFileWriter(const char* path)
      : file_(std::fopen(path, "wb")) {}

  ~DiskFileWriter() override {
    if (file_)
      std::fclose(file_);
  }

  bool ok() const { return file_ != nullptr; }

  std::size_t fwrite(const void* buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fwrite(buffer, 1, size, file_);
  }

private:
  std::FILE* file_;
};

// FileSystem that resolves all paths relative to a base directory on disk.
//
// Usage:
//   hip::DiskFileSystem fs("/path/to/model/dir");
//   hip_compile_with_fs(mlir_data, size, "model.dll", nullptr, &err, &fs);
//   // constants.bin is written to /path/to/model/dir/constants.bin
//
class DiskFileSystem : public morphizen::FileSystem {
public:
  // base_dir: directory under which all filenames are resolved.
  //           Pass "" or "." to use the current working directory.
  explicit DiskFileSystem(const char* base_dir) : base_dir_(base_dir) {
    // Normalise: ensure base_dir ends with a separator.
    if (!base_dir_.empty() && base_dir_.back() != '/' &&
        base_dir_.back() != '\\')
      base_dir_ += '/';
  }

  morphizen::FileReader* create_reader(const char* path) override {
    auto* r = new DiskFileReader(resolve(path).c_str());
    if (!r->ok()) {
      delete r;
      return nullptr;
    }
    return r;
  }

  morphizen::FileWriter* create_writer(const char* path) override {
    auto* w = new DiskFileWriter(resolve(path).c_str());
    if (!w->ok()) {
      delete w;
      return nullptr;
    }
    return w;
  }

  void destroy_reader(morphizen::FileReader* r) override { delete r; }
  void destroy_writer(morphizen::FileWriter* w) override { delete w; }

private:
  std::string resolve(const char* path) const {
    // Absolute paths are used as-is.
    if (path && (path[0] == '/' || path[0] == '\\' ||
                 (std::strlen(path) > 2 && path[1] == ':')))
      return path;
    return base_dir_ + (path ? path : "");
  }

  std::string base_dir_;
};

} // namespace hip
