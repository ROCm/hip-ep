/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <iostream>
#include <memory>
#include <morphizen/file_stream.hpp>

namespace morphizen {

/**
 * Temporary file stream using std::iostream interface.
 * Wraps FileStream with tmpfile() backend for automatic cleanup.
 */
class TempFileStream {
public:
  TempFileStream();
  ~TempFileStream() = default;

  std::iostream &get_stream();
  std::ostream &get_write_stream();
  std::istream &get_read_stream();
  size_t get_size();

private:
  std::unique_ptr<FileStream> stream_;
};

} // namespace morphizen
