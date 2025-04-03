#pragma once
#include "vaip/export.h"
#include <cstdio>
#include <iostream>
#include <streambuf>
#include <vector>
namespace vaip_core {
class FileBuf : public std::streambuf {

public:
  VAIP_DLL_SPEC explicit FileBuf(FILE* file, std::size_t bufferSize = 4096);
  virtual ~FileBuf();
  // Handles reading from FILE*
  virtual int_type underflow() override final;
  // Handles writing to FILE*
  virtual int_type overflow(int_type ch) override;
  // Flushes the output buffer
  virtual int sync() override { return flushBuffer() ? 0 : -1; }

  // Seek support using fseek
  std::streampos seekoff(std::streamoff offset, std::ios_base::seekdir way,
                         std::ios_base::openmode which) override;

  std::streampos seekpos(std::streampos pos,
                         std::ios_base::openmode which) override;

private:
  bool flushBuffer();

  FILE* file_;
  std::size_t bufferSize_;
  std::vector<char> buffer_;
};

// Utility class for stream interface
class FileStream : public std::iostream {
public:
  VAIP_DLL_SPEC explicit FileStream(FILE* file, size_t bufferSize = 4096);

private:
  FileBuf buf_;
};
} // namespace vaip_core
