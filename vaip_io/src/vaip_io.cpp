/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
#endif
#ifdef _WIN32
#  define fseek64 _fseeki64
#  define ftell64 _ftelli64
#else
#  define fseek64 fseeko
#  define ftell64 ftello
#endif
#include "morphizen/vaip_io.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <vector>

namespace vaip_core {

// imp
struct TempFileHolder {
  TempFileHolder(std::shared_ptr<TempFile> file) : file_{file} {}
  std::shared_ptr<TempFile> file_;
};
class FileStreamReader : public IStreamReader, TempFileHolder {
public:
  FileStreamReader(FILE* file) : TempFileHolder{nullptr}, file_(file) {}
  FileStreamReader(TempFile& file)
      : TempFileHolder{file.shared_from_this()}, file_(file.get_file()) {}

private:
  std::optional<std::vector<char>> read(size_t size_hint) const override final;

private:
  FILE* file_;
};

class FileStreamWriter : public IStreamWriter, TempFileHolder {
public:
  FileStreamWriter(FILE* file) : TempFileHolder{nullptr}, file_(file) {}
  FileStreamWriter(TempFile& file)
      : TempFileHolder{file.shared_from_this()}, file_(file.get_file()) {}

private:
  size_t write(const char* data, size_t size) override final;

private:
  FILE* file_;
};

class ByteStreamWriter : public IStreamWriter {
public:
  ByteStreamWriter(std::vector<char>& bytes) : bytes_(bytes) {}

private:
  size_t write(const char* data, size_t size) override final;

private:
  std::vector<char>& bytes_;
};
class ByteStreamReader : public IStreamReader {
public:
  ByteStreamReader(const void* data, size_t size)
      : bytes_(static_cast<const char*>(data)), size_{size} {}

private:
  std::optional<std::vector<char>> read(size_t size_hint) const override final;

private:
  const char* bytes_;
  size_t size_;
  mutable size_t pos = 0;
};
template <typename T> class StdStreamReader : public IStreamReader {
public:
  StdStreamReader(T&& stream) : stream_(std::move(stream)) {}
  virtual ~StdStreamReader();

private:
  std::optional<std::vector<char>> read(size_t size_hint) const override final;

private:
  T stream_;
};

class OwnedStreamWriter : public IStreamWriter {
public:
  OwnedStreamWriter(std::unique_ptr<std::ostream>&& stream)
      : stream_(std::move(stream)) {}
  virtual ~OwnedStreamWriter();

private:
  size_t write(const char* data, size_t size) override final;

private:
  std::unique_ptr<std::ostream> stream_;
};

class UnownedStreamWriter : public IStreamWriter {
public:
  UnownedStreamWriter(std::ostream& stream) : stream_(stream) {}

private:
  size_t write(const char* data, size_t size) override final;

private:
  std::ostream& stream_;
};

std::unique_ptr<IStreamReader>
IStreamReader ::from_stream(std::unique_ptr<std::istream> stream) {
  return std::make_unique<StdStreamReader<std::unique_ptr<std::istream>>>(
      std::move(stream));
}
std::unique_ptr<IStreamReader>
IStreamReader ::from_shared_stream(std::shared_ptr<std::istream> stream) {
  return std::make_unique<StdStreamReader<std::shared_ptr<std::istream>>>(
      std::move(stream));
}
//
std::unique_ptr<IStreamReader> IStreamReader::from_bytes(const void* data,
                                                         size_t size) {
  return std::make_unique<ByteStreamReader>(data, size);
}

std::unique_ptr<IStreamReader>
IStreamReader::from_bytes(const std::vector<char>& bytes) {
  return std::make_unique<ByteStreamReader>(bytes.data(), bytes.size());
}

std::unique_ptr<IStreamReader>
IStreamReader::from_path(const std::filesystem::path& path) {
  return std::make_unique<StdStreamReader<std::unique_ptr<std::istream>>>(
      std::make_unique<std::ifstream>(path, std::ios::binary));
}
std::unique_ptr<IStreamReader> IStreamReader::from_FILE(FILE* file) {
  return std::make_unique<FileStreamReader>(file);
}
std::unique_ptr<IStreamReader> IStreamReader::from_TempFile(TempFile& file) {
  return std::make_unique<FileStreamReader>(file);
}

std::unique_ptr<IStreamWriter>
IStreamWriter::from_bytes(std::vector<char>& bytes) {
  return std::make_unique<ByteStreamWriter>(bytes);
}
std::unique_ptr<IStreamWriter> IStreamWriter::from_FILE(FILE* file) {
  return std::make_unique<FileStreamWriter>(file);
}
std::unique_ptr<IStreamWriter> IStreamWriter::from_TempFile(TempFile& file) {
  return std::make_unique<FileStreamWriter>(file);
}

std::unique_ptr<IStreamWriter>
IStreamWriter::from_path(const std::filesystem::path& path) {
  return std::make_unique<OwnedStreamWriter>(
      std::make_unique<std::ofstream>(path, std::ios::binary));
}

std::unique_ptr<IStreamWriter>
IStreamWriter::from_owned_ostream(std::unique_ptr<std::ostream> stream) {
  return std::make_unique<OwnedStreamWriter>(std::move(stream));
}

std::unique_ptr<IStreamWriter>
IStreamWriter::from_unowned_ostream(std::ostream& stream) {
  return std::make_unique<UnownedStreamWriter>(stream);
}

std::optional<std::vector<char>>
FileStreamReader::read(size_t size_hint) const {
  auto ret = std::vector<char>();
  ret.resize(size_hint);
  auto read_size =
      std::fread(&ret[0], sizeof(unsigned char), size_hint, this->file_);
  if (read_size == 0) {
    return std::nullopt;
  } else {
    ret.resize(read_size);
  }
  return ret;
}

size_t FileStreamWriter::write(const char* data, size_t size) {
  return std::fwrite(data, sizeof(char), size, this->file_);
}

size_t ByteStreamWriter::write(const char* data, size_t size) {
  if (!data || size == 0) {
    return 0;
  }
  bytes_.insert(bytes_.end(), data, data + size);
  return size;
}

template <typename T> StdStreamReader<T>::~StdStreamReader() {}
template <typename T>
std::optional<std::vector<char>>
StdStreamReader<T>::read(size_t size_hint) const {
  auto ret = std::vector<char>(size_hint);
  auto read_size = stream_->read(ret.data(), size_hint).gcount();
  ret.resize(read_size);
  return ret;
}

OwnedStreamWriter::~OwnedStreamWriter() {}

size_t OwnedStreamWriter::write(const char* data, size_t size) {
  auto ok = stream_->write(data, size).good();
  auto ret = ok ? size : 0;
  return ret;
}

size_t UnownedStreamWriter::write(const char* data, size_t size) {
  auto ok = stream_.write(data, size).good();
  auto ret = ok ? size : 0;
  return ret;
}

std::optional<std::vector<char>>
ByteStreamReader::read(size_t size_hint) const {
  if (pos >= size_) {
    return std::nullopt;
  }
  auto ret = std::vector<char>();
  if (pos + size_hint < size_) {
    ret.resize(size_hint);
    std::memcpy(ret.data(), bytes_ + pos, size_hint);
    pos = pos + size_hint;
  } else {
    auto read_size = size_ - pos;
    ret.resize(read_size);
    std::memcpy(ret.data(), bytes_ + pos, read_size);
    pos = pos + read_size;
  }
  return ret;
}

void stream_copy(const IStreamReader& src, IStreamWriter& dst,
                 size_t size_hint) {
  for (auto buf = src.read(size_hint); buf; buf = src.read(size_hint)) {
    auto write_size = dst.write(buf->data(), buf->size());
    CHECK_EQ(write_size, buf->size());
  }
  return;
}

size_t TempFile::current_position() const { return ftell64(file_); }

void TempFile::reset_position() { CHECK(fseek64(file_, 0, SEEK_SET) == 0); }

TempFile::TempFile() : file_{tmpfile()} { CHECK(file_ != nullptr); }
TempFile::~TempFile() { fclose(file_); };

std::pair<std::unique_ptr<IStreamReader>, size_t> TempFile::build_reader() {
  CHECK(fseek64(file_, 0, SEEK_END) == 0);
  auto size = ftell64(file_);
  CHECK(fseek64(file_, 0, SEEK_SET) == 0);
  return {IStreamReader::from_TempFile(*this), size};
}

std::unique_ptr<IStreamWriter> TempFile::build_writer() {
  CHECK(fseek64(file_, 0, SEEK_END) == 0);
  return IStreamWriter::from_TempFile(*this);
};

} // namespace vaip_core
