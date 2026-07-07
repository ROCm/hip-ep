/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/env_config.hpp"
#include <glog/logging.h>
#include <memory>
#include <morphizen/file_stream.hpp>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= n)

// 64-bit file position helpers.
// On Windows MSVC (LLP64), `long` is 32-bit, so std::ftell / std::fseek can
// only handle files up to 2 GB. Use platform-specific 64-bit variants instead.
#ifdef _WIN32
#define FTELL64(f) _ftelli64(f)
#define FSEEK64(f, o, w) _fseeki64(f, o, w)
#else
#define FTELL64(f) ftello(f)
#define FSEEK64(f, o, w) fseeko(f, o, w)
#endif

namespace morphizen {
FileBuf::FileBuf(FILE *file, std::size_t buffer_size) {
  // Constructor to initialize the file stream buffer
  // and set up the read and write buffers.
  file_ = file;
  if (!file_) {
    throw std::runtime_error("Invalid FILE* provided");
  }
  get_buffer_.resize(buffer_size);
  put_buffer_.resize(
      buffer_size); // overflow() needs one more ch to be written.
  setg(get_buffer_.data(), get_buffer_.data(),
       get_buffer_.data()); // Set read buffer
  setp(
      put_buffer_.data(),
      put_buffer_.data() + put_buffer_.size() -
          1 /*overflow() needs one more ch to be written*/); // Set write buffer
  get_pos_ = FTELL64(file_); // Initialize current position
  put_pos_ = get_pos_;       // Initialize current position
  CHECK_NE(get_pos_, -1) << " cannot ftell";
  MY_LOG(1) << " init done: "
            << "pos_=" << get_pos_ << ";" //
      ;
}

FileBuf::~FileBuf() {
  sync(); // Ensure all data is flushed before destruction
  if (file_) {
    fclose(file_); // Close the file if it was opened
  }
}

std::streambuf::int_type FileBuf::underflow() {
  // Handles reading from FILE*
  auto old_pos = FTELL64(file_);
  if (old_pos == -1) {
    MY_LOG(1) << "conner case, cannot test current file position.";
    return traits_type::eof(); // Error in getting position
  }
  auto restore_old_pos =
      std::shared_ptr<void>(nullptr, [old_pos, this](void * /*p*/) {
        auto r = FSEEK64(file_, old_pos, SEEK_SET);
        CHECK(r == 0) << " conner case: fseek fail";
      });
  auto r = FSEEK64(file_, get_pos_, SEEK_SET);
  CHECK(r == 0) << " conner case: fseek fail";
  std::size_t num_of_element_read = std::fread(
      get_buffer_.data(), sizeof(char_type), get_buffer_.size(), file_);
  if (num_of_element_read == 0) {
    return traits_type::eof();
  }
  get_pos_ = FTELL64(file_);
  CHECK_NE(get_pos_, -1) << "cannot ftell";
  setg(get_buffer_.data(), get_buffer_.data(),
       get_buffer_.data() + num_of_element_read);
  return traits_type::to_int_type(*gptr());
}
std::streambuf::int_type FileBuf::overflow(int_type ch) {
  // Handles writing to FILE*
  auto end = pptr();
  if (ch != traits_type::eof()) {
    *end = (char_type)ch;
    end++; // it is safe, because put_buffer.size() is one more than the buffer
           // size;
  }
  auto old_pos = FTELL64(file_);
  if (old_pos == -1) {
    MY_LOG(1) << "conner case, cannot test current file position.";
    return traits_type::eof(); // Error in getting position
  }
  auto restore_old_pos =
      std::shared_ptr<void>(nullptr, [old_pos, this](void * /*p*/) {
        auto r = FSEEK64(file_, old_pos, SEEK_SET);
        CHECK(r == 0) << " conner case: fseek fail";
      });
  auto r = FSEEK64(file_, put_pos_, SEEK_SET);
  CHECK(r == 0) << " conner case: fseek fail";
  auto num_of_elements = static_cast<std::size_t>(end - pbase());
  auto num_of_element_written =
      std::fwrite(pbase(), sizeof(char_type), num_of_elements, file_);
  CHECK_EQ(num_of_element_written, num_of_elements) << " error writing.";
  setp(put_buffer_.data(),
       put_buffer_.data() + put_buffer_.size() -
           1 /*overflow() needs one more ch to be written*/);
  put_pos_ = FTELL64(file_);
  CHECK_GE(put_pos_, 0);
  return ch; // Typically, ch is returned to indicate success.
}
std::streampos FileBuf::seekoff(std::streamoff offset,
                                std::ios_base::seekdir way,
                                std::ios_base::openmode which) {
  if (which & std::ios_base::in) {
    return seekoff_in(offset, way);
  } else if (which & std::ios_base::out) {
    return seekoff_out(offset, way);
  }
  return -1;
}

std::streampos FileBuf::seekoff_in(std::streamoff offset,
                                   std::ios_base::seekdir way) {
  if (way == std::ios_base::cur) {
    auto num_of_elements = egptr() - gptr();
    if (offset == 0) {                   // tellg
      return get_pos_ - num_of_elements; //
    }
  }
  // Seek support using fseek
  auto new_get_pos = get_pos_;
  if (way == std::ios_base::cur) {
    new_get_pos = new_get_pos + offset;
  } else if (way == std::ios_base::end) {
    auto old_pos = FTELL64(file_);
    if (old_pos == -1) {
      MY_LOG(1) << "conner case, cannot test current file position.";
      return -1;
    }
    auto restore_old_pos =
        std::shared_ptr<void>(nullptr, [old_pos, this](void * /*p*/) {
          auto r = FSEEK64(file_, old_pos, SEEK_SET);
          CHECK(r == 0) << " conner case: fseek fail";
        });
    auto r = FSEEK64(file_, old_pos, SEEK_END);
    CHECK(r == 0) << " conner case: fseek fail";
    new_get_pos = FTELL64(file_);
    CHECK_NE(new_get_pos, -1) << "ftell failed.";
  } else if (way == std::ios_base::beg) {
    new_get_pos = offset;
  }
  CHECK_GE(new_get_pos, 0) << "error";
  get_pos_ = new_get_pos;
  setg(get_buffer_.data(), get_buffer_.data(),
       get_buffer_.data()); // reset read buffer
  return get_pos_;
}

std::streampos FileBuf::seekoff_out(std::streamoff offset,
                                    std::ios_base::seekdir way) {
  if (way == std::ios_base::cur) {
    auto num_of_elements = pptr() - pbase();
    if (offset == 0) {                   // tellp optimization;
      return put_pos_ + num_of_elements; //
    }
  }
  auto new_put_pos = put_pos_;
  if (way == std::ios_base::cur) {
    new_put_pos = new_put_pos + offset;
  } else if (way == std::ios_base::end) {
    auto old_pos = FTELL64(file_);
    if (old_pos == -1) {
      MY_LOG(1) << "conner case, cannot test current file position.";
      return -1;
    }
    auto restore_old_pos =
        std::shared_ptr<void>(nullptr, [old_pos, this](void * /*p*/) {
          auto r = FSEEK64(file_, old_pos, SEEK_SET);
          CHECK(r == 0) << " conner case: fseek fail";
        });
    auto r = FSEEK64(file_, old_pos, SEEK_END);
    CHECK(r == 0) << " conner case: fseek fail";
    new_put_pos = FTELL64(file_);
    CHECK_NE(new_put_pos, -1) << "ftell failed.";
    new_put_pos = new_put_pos + offset;
  } else if (way == std::ios_base::beg) {
    new_put_pos = offset;
  }
  CHECK_GE(new_put_pos, 0) << "error";
  auto flush_ok = flush_buffer();
  CHECK(flush_ok) << " flush fail";
  put_pos_ = new_put_pos;
  CHECK_GE(put_pos_, 0) << "error";
  setp(put_buffer_.data(),
       put_buffer_.data() + put_buffer_.size() -
           1 /*overflow() needs one more ch to be written*/);
  return put_pos_;
}
std::streampos FileBuf::seekpos(std::streampos pos,
                                std::ios_base::openmode which) {
  return seekoff(pos, std::ios_base::beg, which);
}

bool FileBuf::flush_buffer() {
  auto old_pos = FTELL64(file_);
  if (old_pos == -1) {
    MY_LOG(1) << "conner case, cannot test current file position.";
    return false;
  }
  auto restore_old_pos =
      std::shared_ptr<void>(nullptr, [old_pos, this](void * /*p*/) {
        auto r = FSEEK64(file_, old_pos, SEEK_SET);
        CHECK(r == 0) << " conner case: fseek fail";
      });
  auto r = FSEEK64(file_, put_pos_, SEEK_SET);
  CHECK(r == 0) << " conner case: fseek fail";
  std::ptrdiff_t count = pptr() - pbase();
  if (count > 0) {
    std::size_t written = std::fwrite(pbase(), sizeof(char_type), count, file_);
    CHECK_EQ(written, static_cast<std::size_t>(count));
    setp(put_buffer_.data(),
         put_buffer_.data() + put_buffer_.size() -
             1 /*overflow() needs one more ch to be written*/);
    put_pos_ = put_pos_ + written;
    return written == static_cast<std::size_t>(count);
  }
  return true;
}
FileStream::FileStream(FILE *file, size_t bufferSize)
    : std::iostream(&buf_), buf_(file, bufferSize) {
  // Constructor to initialize the FileStream with a FILE*
  if (!file) {
    throw std::runtime_error("Invalid FILE* provided");
  }
}
} // namespace morphizen
