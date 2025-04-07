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

#pragma once
#include "./vaip.hpp"
#include <cstring>
#include <memory>
#include <vector>

namespace vaip_core {
template <typename T> using unique_with_del = std::unique_ptr<T, void (*)(T*)>;

template <typename T> unique_with_del<T> empty_uniq() {
  return unique_with_del<T>(nullptr, [](T*) {});
}

template <typename T> unique_with_del<T> uniq(T* t) {
  return unique_with_del<T>(t, [](T* ptr) { delete ptr; });
}

class MetaIOReader {
public:
  virtual ~MetaIOReader() = default;
  virtual void sequential_read(char* buffer, size_t size) const = 0;
  virtual void rewind() = 0;
  virtual size_t size() const = 0;
};
// Used for replacing a vector to save space
class MetaIOBuffer {
public:
  virtual ~MetaIOBuffer() = default;
  virtual void read(char* buffer, size_t offset, size_t size) const = 0;
  virtual void write(const char* buffer, size_t offset, size_t size) = 0;
  virtual size_t size() const = 0;
};
// For sequential save only
class MetaIOWriter {
public:
  virtual ~MetaIOWriter() = default;
  virtual void sequential_write(const char* buffer, size_t size) = 0;
};

class MetaIOAPI {
public:
  static void io_copy_to(const MetaIOReader& src, MetaIOWriter& dst) {
    constexpr size_t buffer_size = 64 * 1024;
    char* buffer =
        new char[buffer_size]; // 64KB, on heap to avoid crash the stack
    size_t size = src.size();
    for (size_t i = 0; i < size; i += buffer_size) {
      size_t remain_size = std::min(size - i, buffer_size);
      src.sequential_read(buffer, remain_size);
      dst.sequential_write(buffer, remain_size);
    }
    delete[] buffer;
  }

  virtual ~MetaIOAPI() = default;
  virtual unique_with_del<MetaIOReader>
  to_reader(unique_with_del<MetaIOBuffer> buffer) = 0;
  virtual unique_with_del<MetaIOBuffer> open_buffer() = 0;
  virtual unique_with_del<MetaIOWriter> open_writer(const char* filename) = 0;
  virtual unique_with_del<MetaIOReader> open_read(const char* filename) = 0;
};

class IOAPI : public MetaIOAPI {
public:
  IOAPI(std::shared_ptr<PassContext> ctx);
  virtual ~IOAPI() override final;
  virtual unique_with_del<MetaIOReader>
  to_reader(unique_with_del<MetaIOBuffer> buffer) override final;
  virtual unique_with_del<MetaIOBuffer> open_buffer() override final;
  virtual unique_with_del<MetaIOWriter>
  open_writer(const char* filename) override final;
  virtual unique_with_del<MetaIOReader>
  open_read(const char* filename) override final;

private:
  std::shared_ptr<PassContext> ctx_;
};

}; // namespace vaip_core
