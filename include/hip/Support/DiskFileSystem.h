/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen-foundation/file_io.hpp"
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#ifdef _WIN32
#include <io.h>
#endif

namespace mlir {
namespace hip {

using FilePtr = std::unique_ptr<std::FILE, decltype(&std::fclose)>;

inline int fileSeek(std::FILE *f, int64_t offset, int origin) {
#ifdef _WIN32
  return _fseeki64(f, offset, origin);
#else
  return fseeko(f, offset, origin);
#endif
}

inline int64_t fileTell(std::FILE *f) {
#ifdef _WIN32
  return _ftelli64(f);
#else
  return ftello(f);
#endif
}

class DiskFileReader : public morphizen::FileReader {
public:
  explicit DiskFileReader(const char *path)
      : file_(std::fopen(path, "rb"), &std::fclose), size_(0) {
    if (!file_)
      return;
    fileSeek(file_.get(), 0, SEEK_END);
    auto pos = fileTell(file_.get());
    if (pos < 0) {
      file_.reset();
      return;
    }
    size_ = static_cast<std::size_t>(pos);
    fileSeek(file_.get(), 0, SEEK_SET);
  }

  bool ok() const { return file_ != nullptr; }
  std::size_t size() const override { return size_; }

  void rewind() const override {
    if (file_)
      fileSeek(file_.get(), 0, SEEK_SET);
  }

  std::size_t fread(void *buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fread(buffer, 1, size, file_.get());
  }

private:
  FilePtr file_;
  std::size_t size_;
};

class DiskFileWriter : public morphizen::FileWriter {
public:
  explicit DiskFileWriter(const char *path)
      : file_(std::fopen(path, "wb"), &std::fclose) {}

  bool ok() const { return file_ != nullptr; }

  std::size_t fwrite(const void *buffer, std::size_t size) const override {
    if (!file_)
      return 0;
    return std::fwrite(buffer, 1, size, file_.get());
  }

private:
  FilePtr file_;
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

} // namespace hip
} // namespace mlir
