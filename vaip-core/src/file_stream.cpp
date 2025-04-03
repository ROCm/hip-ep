#include "./file_stream.hpp"

namespace vaip_core {
FileBuf::FileBuf(FILE* file, std::size_t bufferSize) {
  // Constructor to initialize the file stream buffer
  // and set up the read and write buffers.
  file_ = file;
  bufferSize_ = bufferSize;
  if (!file_) {
    throw std::runtime_error("Invalid FILE* provided");
  }
  buffer_.resize(bufferSize_ + 1);    // Extra space for safety
  setg(buffer_.data(), buffer_.data(),
       buffer_.data());               // Set read buffer
  setp(buffer_.data(),
       buffer_.data() + bufferSize_); // Set write buffer
}

FileBuf::~FileBuf() {
  sync();          // Ensure all data is flushed before destruction
  if (file_) {
    fclose(file_); // Close the file if it was opened
  }
}

std::streambuf::int_type FileBuf::underflow() {
  // Handles reading from FILE*
  if (gptr() < egptr()) {
    return traits_type::to_int_type(*gptr());
  }
  std::size_t bytesRead = std::fread(buffer_.data(), 1, bufferSize_, file_);
  if (bytesRead == 0) {
    return traits_type::eof();
  }
  setg(buffer_.data(), buffer_.data(), buffer_.data() + bytesRead);
  return traits_type::to_int_type(*gptr());
}
std::streambuf::int_type FileBuf::overflow(int_type ch) {
  // Handles writing to FILE*
  if (ch != traits_type::eof()) {
    *pptr() = traits_type::to_char_type(ch);
    pbump(1);
  }
  return flushBuffer() ? ch : traits_type::eof();
}
std::streampos FileBuf::seekoff(std::streamoff offset,
                                std::ios_base::seekdir way,
                                std::ios_base::openmode which) {
  // Seek support using fseek
  if (which & std::ios_base::in) {
    if (way == std::ios_base::cur) {
      offset += gptr() - eback();
    } else if (way == std::ios_base::end) {
      std::fseek(file_, 0, SEEK_END);
      offset += std::ftell(file_);
    }
    std::fseek(file_, (long)offset, SEEK_SET);
    return std::ftell(file_);
  } else if (which & std::ios_base::out) {
    flushBuffer();
    if (way == std::ios_base::cur) {
      offset += pptr() - pbase();
    } else if (way == std::ios_base::end) {
      std::fseek(file_, 0, SEEK_END);
      offset += std::ftell(file_);
    }
    std::fseek(file_, (long)offset, SEEK_SET);
    return std::ftell(file_);
  }
  return -1;
}
std::streampos FileBuf::seekpos(std::streampos pos,
                                std::ios_base::openmode which) {
  return seekoff(pos, std::ios_base::beg, which);
}

bool FileBuf::flushBuffer() {
  std::ptrdiff_t count = pptr() - pbase();
  if (count > 0) {
    std::size_t written = std::fwrite(pbase(), 1, count, file_);
    pbump(-static_cast<int>(count)); // Reset write position
    return written == static_cast<std::size_t>(count);
  }
  return true;
}
FileStream::FileStream(FILE* file, size_t bufferSize)
    : std::iostream(&buf_), buf_(file, bufferSize) {
  // Constructor to initialize the FileStream with a FILE*
  if (!file) {
    throw std::runtime_error("Invalid FILE* provided");
  }
}
} // namespace vaip_core
