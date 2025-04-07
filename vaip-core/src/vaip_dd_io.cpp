/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "morphizen/vaip_dd_io.hpp"
#include "glog/logging.h"

#ifdef _WIN32
#  define fseek64 _fseeki64
#  define ftell64 _ftelli64
#else
#  define fseek64 fseeko
#  define ftell64 ftello
#endif

namespace vaip_core {

class IOReader : public MetaIOReader {
public:
  virtual ~IOReader();
  IOReader(std::unique_ptr<CacheFileReader> reader);
  virtual void rewind() override final;
  virtual void sequential_read(char* buffer, size_t size) const override final;
  virtual size_t size() const override final;

private:
  std::unique_ptr<CacheFileReader> reader_;
};

class IOWriter : public MetaIOWriter {
public:
  IOWriter(std::unique_ptr<CacheFileWriter> writer);
  virtual ~IOWriter();
  virtual void sequential_write(const char* buffer, size_t size) override final;

private:
  std::unique_ptr<CacheFileWriter> writer_;
};

class Tmp : public MetaIOBuffer {
public:
  virtual ~Tmp() override final;
  Tmp(FILE* file);
  virtual void read(char* buffer, size_t offset,
                    size_t size) const override final;
  virtual void write(const char* buffer, size_t offset,
                     size_t size) override final;
  virtual size_t size() const override final;
  FILE* release_file();

private:
  FILE* file_;
};

class TmpReader : public MetaIOReader {
public:
  virtual ~TmpReader();
  TmpReader(FILE* file);
  virtual void rewind() override final;
  virtual void sequential_read(char* buffer, size_t size) const override final;
  virtual size_t size() const override final;

private:
  FILE* file_;
};

static size_t file_size(const FILE* file) {
  auto f = const_cast<FILE*>(file);
  auto old_offset = ftell64(f);
  fseek64(f, 0, SEEK_END);
  auto size = ftell64(f);
  fseek64(f, old_offset, SEEK_SET);
  return size;
}

TmpReader::TmpReader(FILE* file) { file_ = file; }
TmpReader::~TmpReader() { fclose(file_); }

size_t TmpReader::size() const { return file_size(file_); }

void TmpReader::rewind() { fseek64(file_, 0, SEEK_SET); }

void TmpReader::sequential_read(char* buffer, size_t size) const {
  fread(buffer, size, 1, file_);
}

Tmp::Tmp(FILE* file) { file_ = file; }

Tmp::~Tmp() {
  if (file_) {
    fclose(file_);
  }
}

void Tmp::read(char* buffer, size_t offset, size_t size) const {
  fseek64(file_, offset, SEEK_SET);
  fread(buffer, sizeof(char), size, file_);
}

void Tmp::write(const char* buffer, size_t offset, size_t size) {
  fseek64(file_, offset, SEEK_SET);
  fwrite(buffer, 1, size, file_);
}

size_t Tmp::size() const { return file_size(file_); }

FILE* Tmp::release_file() {
  auto ret = file_;
  file_ = nullptr;
  return ret;
}

IOWriter::~IOWriter() {}
IOWriter::IOWriter(std::unique_ptr<CacheFileWriter> writer) {
  writer_ = std::move(writer);
}

void IOWriter::sequential_write(const char* buffer, size_t size) {
  writer_->fwrite(buffer, size);
}

IOReader::~IOReader() {}

IOReader::IOReader(std::unique_ptr<CacheFileReader> reader) {
  reader_ = std::move(reader);
}

void IOReader::sequential_read(char* buffer, size_t size) const {
  reader_->fread(buffer, size);
}

void IOReader::rewind() { reader_->rewind(); }

size_t IOReader::size() const { return reader_->size(); }

IOAPI::~IOAPI() {}

unique_with_del<MetaIOReader>
IOAPI::to_reader(unique_with_del<MetaIOBuffer> buffer) {
  auto buf = reinterpret_cast<Tmp*>(buffer.get());
  auto file = buf->release_file();
  auto ret = new TmpReader(file);
  ret->rewind();
  return uniq<MetaIOReader>(ret);
}

unique_with_del<MetaIOBuffer> IOAPI::open_buffer() {
#if _WIN32
  FILE* tmp_file = nullptr;
  auto err = tmpfile_s(&tmp_file);
  CHECK(err == 0) << "tmpfile_s error";
#else
  FILE* tmp_file = tmpfile();
  CHECK(tmp_file != nullptr) << "tmpfile error";
#endif
  return uniq<MetaIOBuffer>(new Tmp(tmp_file));
}

unique_with_del<MetaIOReader> IOAPI::open_read(const char* filename) {
  auto reader = ctx_->open_file_for_read(filename);
  auto ret = new IOReader(std::move(reader));
  return uniq<MetaIOReader>(ret);
}

unique_with_del<MetaIOWriter> IOAPI::open_writer(const char* filename) {
  auto writer = ctx_->open_file_for_write(filename);
  auto ret = new IOWriter(std::move(writer));
  return uniq<MetaIOWriter>(ret);
}

IOAPI::IOAPI(std::shared_ptr<PassContext> ctx) { ctx_ = ctx; }
}; // namespace vaip_core
