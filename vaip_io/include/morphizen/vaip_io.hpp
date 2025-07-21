/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vaip_core {
class TempFile;
class IStreamReader {
public:
  IStreamReader() = default;
  virtual ~IStreamReader() = default;
  virtual std::optional<std::vector<char>>
  read(size_t size_hint = 1024u * 8u) const = 0;

public:
  static std::unique_ptr<IStreamReader>
  from_stream(std::unique_ptr<std::istream> stream);
  static std::unique_ptr<IStreamReader>
  from_shared_stream(std::shared_ptr<std::istream> stream);
  static std::unique_ptr<IStreamReader> from_bytes(const void* data,
                                                   size_t size);
  static std::unique_ptr<IStreamReader> from_bytes(const std::vector<char>&);
  static std::unique_ptr<IStreamReader> from_FILE(FILE*);
  static std::unique_ptr<IStreamReader>
  from_path(const std::filesystem::path& path);
  static std::unique_ptr<IStreamReader> from_TempFile(TempFile&);
};
class IStreamWriter {
public:
  IStreamWriter() = default;
  virtual ~IStreamWriter() = default;
  virtual size_t write(const char* data, size_t size) = 0;

public:
  static std::unique_ptr<IStreamWriter> from_bytes(std::vector<char>&);
  static std::unique_ptr<IStreamWriter> from_FILE(FILE*);
  static std::unique_ptr<IStreamWriter> from_TempFile(TempFile&);

  static std::unique_ptr<IStreamWriter>
  from_path(const std::filesystem::path& path);
  static std::unique_ptr<IStreamWriter>
  from_owned_ostream(std::unique_ptr<std::ostream> stream);
  static std::unique_ptr<IStreamWriter>
  from_unowned_ostream(std::ostream& stream);
};

void stream_copy(const IStreamReader& src, IStreamWriter& dst,
                 size_t size_hint = 1024u * 8u);

class IStreamWriterBuilder {
public:
  virtual std::unique_ptr<IStreamWriter> build(const std::string&) = 0;
};
class TempFile : public std::enable_shared_from_this<TempFile> {
public:
  TempFile();
  TempFile(const TempFile&) = delete;
  ~TempFile();

  std::pair<std::unique_ptr<IStreamReader>, size_t> build_reader();
  std::unique_ptr<IStreamWriter> build_writer();
  size_t current_position() const;
  void reset_position();
  FILE* get_file() const { return file_; }

private:
  FILE* file_;
};
template <typename F, typename... Args>
inline std::unique_ptr<IStreamReader>
stream_filter(const IStreamReader& src, const F& filter, Args&&... args) {
  auto temp_file = std::make_shared<TempFile>();
  auto writer = temp_file->build_writer();
  filter(src, *writer, std::forward<Args>(args)...);
  auto reader_and_size = temp_file->build_reader();
  return std::move(reader_and_size.first);
}
} // namespace vaip_core
