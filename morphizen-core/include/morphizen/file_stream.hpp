/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/export.h"
#include <cstdio>
#include <iostream>
#include <streambuf>
#include <vector>
namespace morphizen {
class FileBuf : public std::streambuf {

public:
  MORPHIZEN_DLL_SPEC explicit FileBuf(FILE* file,
                                      std::size_t bufferSize = 4 * 1024 * 1024);
  virtual ~FileBuf();
  // Handles reading from FILE*
  virtual int_type underflow() override final;
  // Handles writing to FILE*
  virtual int_type overflow(int_type ch) override;
  // Flushes the output buffer
  virtual int sync() override { return flush_buffer() ? 0 : -1; }

  // Seek support using fseek
  std::streampos seekoff(std::streamoff offset, std::ios_base::seekdir way,
                         std::ios_base::openmode which) override;
  std::streampos seekoff_in(std::streamoff offset, std::ios_base::seekdir way);
  std::streampos seekoff_out(std::streamoff offset, std::ios_base::seekdir way);
  std::streampos seekpos(std::streampos pos,
                         std::ios_base::openmode which) override;

private:
  bool flush_buffer();

  FILE* file_;
  std::streambuf::off_type get_pos_ = 0;
  std::streambuf::off_type put_pos_ = 0;
  std::vector<char_type> get_buffer_;
  std::vector<char_type> put_buffer_;
};

// Utility class for stream interface
class FileStream : public std::iostream {
public:
  MORPHIZEN_DLL_SPEC explicit FileStream(FILE* file,
                                         size_t bufferSize = 4 * 1024 * 1024);

private:
  FileBuf buf_;
};
} // namespace morphizen
