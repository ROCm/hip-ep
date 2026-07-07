/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/export.h"
#include <cstdio>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <streambuf>
#include <vector>
namespace morphizen {
template <typename T> class MemBuffer : public std::streambuf {
public:
  static std::unique_ptr<MemBuffer<T>>
  create(const void* base, std::size_t size, std::unique_ptr<T>&& owner) {
    return std::make_unique<MemBuffer<T>>(base, size, std::move(owner));
  }

public:
  explicit MemBuffer(const void* base, std::size_t size,
                     std::unique_ptr<T> owner)
      : base_((const char*)base), size_(size), owner_(std::move(owner)) {
    my_setg(base_);
    // disable put
    setp((char_type*)base_, (char_type*)base_);
  }

  virtual ~MemBuffer() {}
  virtual int_type underflow() override final {
    if (gptr() >= base_ + size_) {
      return traits_type::eof();
    }
    my_setg(gptr());
    return traits_type::to_int_type(*gptr());
  }
  virtual int_type overflow(int_type /*ch*/) override {
    return traits_type::eof();
  }

  std::streampos seekoff(std::streamoff offset, std::ios_base::seekdir way,
                         std::ios_base::openmode which) override {
    if (which & std::ios_base::in) {
      if (way == std::ios_base::beg) {
        my_setg(base_ + offset);
      } else if (way == std::ios_base::cur) {
        my_setg(gptr() + offset);
      } else if (way == std::ios_base::end) {
        my_setg(base_ + size_ + offset);
      }
      return gptr() - base_;
    } else {
      // CHECK(false) << " do not support writing";
      return -1;
    }
  }

  std::streampos seekpos(std::streampos pos,
                         std::ios_base::openmode which) override {
    // optionally call seekoff() here to centralize logic
    return seekoff(pos, std::ios_base::beg, which);
  }
  const char* base() const { return base_; }

private:
  static constexpr size_t PAGE_SIZE = 4096u;
  static constexpr uintptr_t PAGE_SIZE_MASK = PAGE_SIZE - 1;
  void my_setg(const char* cur) {
    uintptr_t beg = ((uintptr_t)cur) & ~PAGE_SIZE_MASK;
    /* setg() // set pointers for read buffer
     *_IGfirst = _First;
     *_IGnext  = _Next;
     *_IGcount = static_cast<int>(_Last - _Next);
     */
    // we see msvc have above limitation, the max end must be less than
    // std::numeric_limits<int>::max(), otherwise , *_IGcount will be negative
    // which results in undefined bahavior
    uintptr_t end_max = ((uintptr_t)base_) +
                        static_cast<uintptr_t>(std::numeric_limits<int>::max());
    uintptr_t end_intented = beg + size_;
    uintptr_t end = std::min(end_intented, end_max);
    setg((char*)beg, (char*)cur, (char*)end);
  }

private:
  const char* base_;

  size_t size_;
  std::unique_ptr<T> owner_;
};

template <typename T> class MemStream : public std::iostream {
public:
  static std::unique_ptr<MemStream<T>>
  create(size_t size, std::shared_ptr<MemBuffer<T>> sb) {
    return std::make_unique<MemStream<T>>(size, std::move(sb));
  }
  MemStream(std::shared_ptr<MemBuffer<T>> sb)
      : std::iostream(sb.get()), buffer_{sb} {}

public:
  const char* offset(std::streamoff offset) const {
    return buffer_->base() + offset;
  }

private:
  // a buffer can be used by multiple streams
  std::shared_ptr<MemBuffer<T>> buffer_;
};

/**
 * Write-only streambuf for std::vector<char>.
 * Replaces IStreamWriter::from_bytes().
 */
class MemoryOutputStreambuf : public std::streambuf {
public:
  explicit MemoryOutputStreambuf(std::vector<char>& buffer) : buffer_(buffer) {}

protected:
  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      buffer_.push_back(static_cast<char>(c));
    }
    return c;
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override {
    buffer_.insert(buffer_.end(), s, s + n);
    return n;
  }

private:
  std::vector<char>& buffer_;
};

} // namespace morphizen
