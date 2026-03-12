/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen-foundation/file_io.hpp"
#include <cstdio>
#include <filesystem>
#include <string>
#ifdef _WIN32
#include <io.h>
#endif

namespace mlir::hip {

class DiskFileReader : public morphizen::FileReader {
public:
  explicit DiskFileReader(const char *path)
      : file_(std::fopen(path, "rb")), size_(0) {
    if (!file_)
      return;
#ifdef _WIN32
    _fseeki64(file_, 0, SEEK_END);
    size_ = static_cast<std::size_t>(_ftelli64(file_));
    _fseeki64(file_, 0, SEEK_SET);
#else
    fseeko(file_, 0, SEEK_END);
    size_ = static_cast<std::size_t>(ftello(file_));
    fseeko(file_, 0, SEEK_SET);
#endif
  }

  ~DiskFileReader() override {
    if (file_)
      std::fclose(file_);
  }

  bool ok() const { return file_ != nullptr; }
  std::size_t size() const override { return size_; }

  void rewind() const override {
    if (file_) {
#ifdef _WIN32
      _fseeki64(file_, 0, SEEK_SET);
#else
      fseeko(file_, 0, SEEK_SET);
#endif
    }
  }

  std::size_t fread(void *buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fread(buffer, 1, size, file_);
  }

private:
  std::FILE *file_;
  std::size_t size_;
};

class DiskFileWriter : public morphizen::FileWriter {
public:
  explicit DiskFileWriter(const char *path) : file_(std::fopen(path, "wb")) {}

  ~DiskFileWriter() override {
    if (file_)
      std::fclose(file_);
  }

  bool ok() const { return file_ != nullptr; }

  std::size_t fwrite(const void *buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fwrite(buffer, 1, size, file_);
  }

private:
  std::FILE *file_;
};

class DiskFileSystem : public morphizen::FileSystem {
public:
  explicit DiskFileSystem(const char *base_dir)
      : base_dir_(std::filesystem::path(base_dir).lexically_normal()) {}

  morphizen::FileReader *create_reader(const char *path) override {
    std::string full_path = (base_dir_ / path).string();
    auto *r = new DiskFileReader(full_path.c_str());
    if (!r->ok()) {
      delete r;
      return nullptr;
    }
    return r;
  }

  morphizen::FileWriter *create_writer(const char *path) override {
    std::string full_path = (base_dir_ / path).string();
    auto *w = new DiskFileWriter(full_path.c_str());
    if (!w->ok()) {
      delete w;
      return nullptr;
    }
    return w;
  }

  void destroy_reader(morphizen::FileReader *r) override { delete r; }
  void destroy_writer(morphizen::FileWriter *w) override { delete w; }

private:
  std::filesystem::path base_dir_;
};

} // namespace mlir::hip
