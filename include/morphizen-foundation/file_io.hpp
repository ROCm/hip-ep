/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace morphizen {

class FileReader {
public:
  FileReader() = default;
  FileReader(const FileReader &) = delete;
  FileReader &operator=(const FileReader &) = delete;
  virtual ~FileReader() = default;

  virtual size_t size() const = 0;
  virtual void rewind() const = 0;
  virtual std::size_t fread(void *buffer, std::size_t size) const = 0;
  virtual void *mmap() { return nullptr; }
};

class FileWriter {
public:
  FileWriter() = default;
  FileWriter(const FileWriter &) = delete;
  FileWriter &operator=(const FileWriter &) = delete;
  virtual ~FileWriter() = default;

  virtual std::size_t fwrite(const void *buffer, std::size_t size) const = 0;
};

class FileSystem {
public:
  FileSystem() = default;
  FileSystem(const FileSystem &) = delete;
  FileSystem &operator=(const FileSystem &) = delete;
  virtual ~FileSystem() = default;

  virtual FileReader *create_reader(const char *path) = 0;
  virtual FileWriter *create_writer(const char *path) = 0;
  virtual void destroy_reader(FileReader *reader) = 0;
  virtual void destroy_writer(FileWriter *writer) = 0;

  template <typename T> struct Deleter {
    FileSystem *fs;
    void operator()(T *ptr) const {
      if constexpr (std::is_same_v<T, FileReader>)
        fs->destroy_reader(ptr);
      else
        fs->destroy_writer(ptr);
    }
  };

  std::unique_ptr<FileReader, Deleter<FileReader>>
  create_reader_template(const char *path) {
    return {create_reader(path), {this}};
  }

  std::unique_ptr<FileWriter, Deleter<FileWriter>>
  create_writer_template(const char *path) {
    return {create_writer(path), {this}};
  }
};

} // namespace morphizen
