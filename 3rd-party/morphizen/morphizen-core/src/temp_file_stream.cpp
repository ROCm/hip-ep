/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <glog/logging.h>
#include <morphizen/temp_file_stream.hpp>
#include <morphizen/util.hpp>

namespace morphizen {

TempFileStream::TempFileStream() {
  FILE *file = create_tmpfile();
  CHECK(file != nullptr) << "Failed to create temporary file";

  // FileStream takes ownership and will close file in destructor
  stream_ = std::make_unique<FileStream>(file);
}

size_t TempFileStream::get_size() {
  stream_->flush();
  auto pos = stream_->tellp();
  stream_->seekp(0, std::ios::end);
  auto size = stream_->tellp();
  stream_->seekp(pos);
  return static_cast<size_t>(size);
}

std::istream &TempFileStream::get_read_stream() {
  stream_->flush();
  stream_->seekg(0, std::ios::beg);
  return *stream_;
}

std::ostream &TempFileStream::get_write_stream() { return *stream_; }

std::iostream &TempFileStream::get_stream() { return *stream_; }

} // namespace morphizen
