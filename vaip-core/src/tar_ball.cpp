/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "tar_ball.hpp"
#include <cstdio>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
// clang-format off
#define DIR void
#ifdef _WIN32
#else
struct stat {};
#endif
#include "tar.h"
// clang-format on
#include <chrono>
#include <cstdint>
#include <string.h>
#include <string>
#include <type_traits>
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

#define BLOCKSIZE 512
union block {
  char buffer[BLOCKSIZE];
  HD_USTAR header;
};
namespace vaip_core {
#define EIGHT_SPACE "        "
#define safe_sprintf(a, fmt, c) snprintf(a, sizeof(a), fmt, c)

static void my_strncpy(char* dst, const char* src, size_t len) {
  // always get a strange error L"Buffer is too small"
  // CHECK(strncpy_s(dst, len, src, len+1) == 0);
  for (auto i = 0u; i < len; ++i) {
    dst[i] = src[i];
  }
}
/**
 * @brief Calculates and verifies the TAR header checksum
 *
 * This function computes the checksum of a TAR header according to the POSIX
 * TAR specification and compares it with the stored checksum to verify the
 * integrity of the TAR header. It supports both POSIX and Sun tar checksum
 * variations.
 *
 * @param header Pointer to the TAR header
 * @return int Checksum validation result:
 *         - 1  : Checksum is valid (computed value matches the stored value)
 *         - -1 : Checksum is invalid (computed value does not match, possibly a
 * corrupted TAR file)
 *         - 0  : The TAR header might be empty (unlikely but possible)
 */
int tar_checksum(block* header) {
  int unsigned_sum = 0; /* the POSIX one :-) */
  int signed_sum = 0;   /* the Sun one :-( */
  int recorded_sum;
  char* p = header->buffer;

  for (auto i = sizeof *header; i-- != 0;) {
    unsigned_sum += (unsigned char)*p;
    signed_sum += (signed char)(*p++);
  }

  if (unsigned_sum == 0)
    return 0;

  /* Adjust checksum to count the "chksum" field as blanks.  */

  for (auto i = sizeof header->header.chksum; i-- != 0;) {
    unsigned_sum -= (unsigned char)header->header.chksum[i];
    signed_sum -= (signed char)(header->header.chksum[i]);
  }
  unsigned_sum += (int)(' ' * sizeof header->header.chksum);
  signed_sum += (int)(' ' * sizeof header->header.chksum);

  auto parsed_sum = std::stoll(header->header.chksum, nullptr, 8);
  if (parsed_sum < 0)
    return -1;

  recorded_sum = (int)parsed_sum;

  if (unsigned_sum != recorded_sum && signed_sum != recorded_sum)
    return -1;

  return 1;
}
int TarWriter::write(const IStreamReader& src, const std::string& name) {
  auto pipe = std::make_shared<TempFile>();
  stream_copy(src, *pipe->build_writer());
  auto [reader, size] = pipe->build_reader();
  return write_internal(*reader, name, size);
}
int TarWriter::write_internal(const IStreamReader& src, const std::string& name,
                              size_t size) {
  const size_t BUFFER_SIZE = 512u;
  auto now = std::chrono::system_clock::now();
  std::time_t now_sec = std::chrono::system_clock::to_time_t(now);
  uint64_t mtime = static_cast<uint64_t>(now_sec);
  block block;
  memset(&block.buffer[0], 0, sizeof(block.buffer));
  static_assert(sizeof(block) == 512);
  static_assert(sizeof(EIGHT_SPACE) == 9);
  auto& header = block.header;
  char typeflag = '0';
  bool is_long_name = name.size() >= sizeof(header.name);
  if (!is_long_name) {
    typeflag = '0';
    my_strncpy(header.name, name.c_str(), name.size());
  } else {
    typeflag = 'L';
    my_strncpy(header.name, "././@LongLink", 13);
  }
  my_strncpy(header.mode, "0000644", 7);
  my_strncpy(header.uid, "0006717", 7);
  my_strncpy(header.gid, "0000112", 7);
  if (!is_long_name) {
    safe_sprintf(header.size, "%011lo", (unsigned long)size);
  } else {
    safe_sprintf(header.size, "%011lo", (unsigned long)name.size());
  }
  safe_sprintf(header.mtime, "%011llo", (long long unsigned int)mtime);
  my_strncpy(header.chksum, EIGHT_SPACE, 8);
  header.typeflag = typeflag;
  my_strncpy(header.magic, TMAGIC, strlen(TMAGIC));
  my_strncpy(header.uname, "abcdefg", 7);
  // my_strncpy(header.gname, "hijklmn", 7);
  unsigned int checksum_value = 0;
  for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
    checksum_value += (uint8_t)block.buffer[i];
  }
  safe_sprintf(header.chksum, "%06o", checksum_value);
  header.chksum[7] = ' ';
  auto t = tarball_.write(&block.buffer[0], sizeof(block));
  CHECK(sizeof(block) == t)
      << "failed to write header. name = " << name << " size = " << size;
  if (name.size() >= sizeof(header.name)) {
    auto size_1 = name.size();
    CHECK(size_1 == tarball_.write(name.data(), size_1))
        << "failed to write data. name = " << name << " size = " << size_1;
    auto const padding_size{512u - static_cast<unsigned int>(size_1 % 512)};
    const char padding_data[512] = {0};
    if (padding_size != 512) {
      CHECK(padding_size == tarball_.write(&padding_data[0], padding_size))
          << "failed to write padding. name = " << name
          << " size = " << padding_size;
    }
    // write header again
    my_strncpy(header.name, name.c_str(), sizeof(header.name));
    my_strncpy(header.chksum, EIGHT_SPACE, 8);
    header.typeflag = '0';
    safe_sprintf(header.size, "%011lo", (unsigned long)size_1);
    unsigned int checksum_value_1 = 0;
    for (unsigned int i = 0; i != sizeof(block.buffer); ++i) {
      checksum_value_1 += (uint8_t)block.buffer[i];
    }
    safe_sprintf(header.chksum, "%06o", checksum_value_1);
    header.chksum[7] = ' ';
    CHECK(sizeof(block) == tarball_.write(&block.buffer[0], sizeof(block)))
        << "failed to write header. name = " << name << " size = " << size_1;
  }
  if (size == 0) {
    return 0;
  }
  for (auto i = 0u; i < size; i += BUFFER_SIZE) {
    auto buffer = src.read(BUFFER_SIZE);
    CHECK(buffer.has_value()) << "failed to read file";
    auto read_size = buffer->size();
    CHECK(read_size > 0);
    CHECK(read_size <= BUFFER_SIZE);
    if (read_size < BUFFER_SIZE) {
      buffer->resize(BUFFER_SIZE);
      // buffer->resize() pad it automatically.
      // auto padding_size =
      //     BUFFER_SIZE - static_cast<unsigned int>(read_size % BUFFER_SIZE);
      // memset(buffer->data() + read_size, 0, padding_size);
      CHECK_GE(i + BUFFER_SIZE, size) << "must be the last trunk";
    }
    CHECK_EQ(BUFFER_SIZE, buffer->size());
    CHECK(tarball_.write((const char*)buffer->data(), buffer->size()))
        << "failed to write data. name = " << name << " size = " << i;
  }
  return 0;
}
TarWriter::~TarWriter() {
  // tar end
  const char padding_data[512] = {0};
  CHECK(tarball_.write(padding_data, 512) == 512) << "dtor tarWriter failed.";
  CHECK(tarball_.write(padding_data, 512) == 512) << "dtor tarWriter failed.";
}

int TarReader::read(IStreamWriterBuilder& dst_builder) {
  const size_t BUFFER_SIZE = 512u;
  std::vector<char> buffer(BUFFER_SIZE, 0);

  auto ret = tarball_.read(sizeof(block));
  if (!ret.has_value()) {
    return 0;
  }
  CHECK(ret->size() == sizeof(block));
  block* block = (union block*)(ret.value().data());
  auto check_ok = tar_checksum(block);
  if (check_ok != 1) {
    return 0;
  }
  CHECK_EQ(check_ok, 1) << "tallball not valid: checksum failed.";
  auto* header = &block->header;
  std::string filename(header->name);
  unsigned long size_ = 0u;
  if (header->typeflag == 'L') {
    size_ = std::stoul(header->size, nullptr, 8);
    filename.resize(size_);
    ret = tarball_.read(size_);
    CHECK(ret.has_value() && ret->size() == size_)
        << "buffer overflow. size_=" << size_;

    auto const padding_size{512u - static_cast<unsigned int>(size_ % 512)};
    if (padding_size != 512) {
      auto tmp_buffer = std::vector<char>(padding_size);
      ret = tarball_.read(padding_size);
      CHECK(ret.has_value() && ret->size() == padding_size)
          << "buffer overflow. size_=" << size_;
    }

    ret = tarball_.read(sizeof(block));
    if (ret.has_value() && ret->size() != sizeof(block)) {
      return 0;
    }
    block = (union block*)(ret.value().data());
    auto check_ok_1 = tar_checksum(block);
    CHECK(check_ok_1) << "tallball not valid: checksum failed.";
    header = &block->header;
  }
  size_ = std::stoul(header->size, nullptr, 8);
  auto dst = dst_builder.build(filename);
  if (size_ == 0) {
    return 1;
  }
  for (auto i = 0u; i < size_; i += BUFFER_SIZE) {
    ret = tarball_.read(BUFFER_SIZE);
    CHECK(ret.has_value() && ret->size() == BUFFER_SIZE)
        << "buffer overflow. name = " << filename << " size_ =" << size_
        << "bytes, read  " << i << "bytes";
    if (i + BUFFER_SIZE <= size_) {
      dst->write(ret->data(), ret->size());
    } else {
      dst->write(ret->data(), size_ % BUFFER_SIZE);
    }
  }
  return 1;
}
} // namespace vaip_core

#include <array>
constexpr bool DEBUG_TAR = false;
#define MY_LOG(...)                                                            \
  do {                                                                         \
    if constexpr (DEBUG_TAR) {                                                 \
      std::cout << "[debug] " << __FILE__ << ":" << __LINE__ << " | ";         \
      (std::cout << __VA_ARGS__) << std::endl;                                 \
    }                                                                          \
  } while (0)

namespace vaip_tar {
template <typename T> T getNextMultiple(T num, T size) {
  if (size <= 0)
    return num;
  T n = num / size;
  return (num % size == 0) ? (num) : ((n + 1) * size);
}
int tar_checksum(block* header) {
  int unsigned_sum = 0; /* the POSIX one :-) */
  int signed_sum = 0;   /* the Sun one :-( */
  int recorded_sum;
  char* p = header->buffer;

  for (auto i = sizeof *header; i-- != 0;) {
    unsigned_sum += (unsigned char)*p;
    signed_sum += (signed char)(*p++);
  }

  if (unsigned_sum == 0)
    return 0;

  /* Adjust checksum to count the "chksum" field as blanks.  */

  for (auto i = sizeof header->header.chksum; i-- != 0;) {
    unsigned_sum -= (unsigned char)header->header.chksum[i];
    signed_sum -= (signed char)(header->header.chksum[i]);
  }
  unsigned_sum += (int)(' ' * sizeof header->header.chksum);
  signed_sum += (int)(' ' * sizeof header->header.chksum);

  auto parsed_sum = std::stoll(header->header.chksum, nullptr, 8);
  if (parsed_sum < 0)
    return -1;

  recorded_sum = (int)parsed_sum;

  if (unsigned_sum != recorded_sum && signed_sum != recorded_sum)
    return -1;

  return 1;
}

// TarFile
TarFile::TarFile(const std::filesystem::path& file_path)
    : file_path(file_path) {
  if (!std::filesystem::exists(file_path)) {
    MY_LOG("file not exists, create empty tar" << file_path);
    return;
  }
  std::ifstream file(file_path);
  if (!file) {
    MY_LOG(std::string("can't open file") + file_path.string());
    return;
  }
  std::vector<char> bytes;
  std::array<char, 512> block_buffer;
  while (!file.eof()) {
    auto readed_size = file.read(block_buffer.data(), sizeof(block)).gcount();
    if (readed_size != sizeof(block)) {
      MY_LOG("attemp read header fail");
      file.close();
      return;
    }
    bytes.insert(bytes.end(), block_buffer.begin(), block_buffer.end());
    block* block = (union block*)(block_buffer.data());
    auto check_ok = tar_checksum(block);
    if (check_ok != 1) {
      MY_LOG(std::string("checksum = ") + std::to_string(check_ok));
      file.close();
      return;
    }
    auto* header = &block->header;
    std::string filename(header->name);
    size_t size_ = 0u;
    if (header->typeflag == 'L') {
      size_ = std::stoul(header->size, nullptr, 8);
      filename.resize(size_);
      readed_size = file.read(filename.data(), size_).gcount();
      if (static_cast<unsigned long>(readed_size) != size_) {
        MY_LOG("attemp read filename fail");
        return;
      }
      bytes.insert(bytes.end(), filename.begin(), filename.end());
      auto const padding_size{512u - static_cast<unsigned int>(size_ % 512)};
      if (padding_size != 512) {
        auto tmp_buffer = std::vector<char>(padding_size);
        readed_size = file.read(tmp_buffer.data(), padding_size).gcount();
        if (readed_size != padding_size) {
          MY_LOG("attemp read padding fail");
          return;
        }
        bytes.insert(bytes.end(), tmp_buffer.begin(), tmp_buffer.end());
      }
      readed_size = file.read(block_buffer.data(), sizeof(block)).gcount();
      if (readed_size != sizeof(block)) {
        MY_LOG("attemp read header 2 fail");
        return;
      }
      bytes.insert(bytes.end(), block_buffer.begin(), block_buffer.end());

      block = (union block*)(block_buffer.data());
      auto check_ok_1 = tar_checksum(block);
      if (check_ok_1 != 1) {
        MY_LOG(std::string("checksum = ") + std::to_string(check_ok_1));
        return;
      }
      header = &block->header;
    }

    size_ = std::stoul(header->size, nullptr, 8);
    if (size_ == 0) {
      MY_LOG("size_ == 0,empty file.");
      this->append(TarEntry::create_from_mem(std::move(bytes)));
      bytes = std::vector<char>();
      MY_LOG(filename + " readed");
      MY_LOG("currently position = " << file.tellg());
      continue;
    }
    size_t need_read_size = getNextMultiple(size_, size_t(512u));
    std::vector<char> temp(need_read_size);
    MY_LOG(std::to_string(file.tellg()));
    readed_size = file.read(temp.data(), need_read_size).gcount();
    if (static_cast<size_t>(readed_size) != need_read_size) {
      MY_LOG(std::string("read data fail,need read ") +
             std::to_string(need_read_size) + ", but only read " +
             std::to_string(readed_size));
      file.close();
      return;
    }
    bytes.insert(bytes.end(), std::make_move_iterator(temp.begin()),
                 std::make_move_iterator(temp.end()));
    this->append(TarEntry::create_from_mem(std::move(bytes)));
    bytes = std::vector<char>();
    MY_LOG(filename + " readed");
    MY_LOG(std::string("currently position = ") + std::to_string(file.tellg()));
  }
  file.close();
}

int TarFile::save(bool overwrite) {
  if (std::filesystem::exists(this->file_path) && !overwrite) {
    MY_LOG(this->file_path.string() + " already exists. skip save");
    return 0;
  }
  std::ofstream file(this->file_path, std::ios::out);
  if (!file) {
    MY_LOG(this->file_path.string() + " can't create file. save failed");
    return 1;
  }
  for (auto entry : this->entries) {
    file.write(entry->entry_data(), entry->entry_size());
  }
  const char padding_data[512] = {0};

  file.write(padding_data, 512);
  file.write(padding_data, 512);
  file.close();
  return 0;
}

int TarFile::append(std::shared_ptr<TarEntry> entry) {
  this->entries.push_back(entry);
  return 0;
}

// tarentry

std::shared_ptr<TarEntry> TarEntry::create_from_mem(std::vector<char>&& data) {
  return std::shared_ptr<TarEntry>(new TarEntry(std::move(data)));
}

std::string TarEntry::get_name() {
  const block* block = (union block*)(this->datas.data());
  bool is_long_name = block->header.typeflag == 'L';
  if (is_long_name) {
    auto name_size = std::stoul(block->header.size, nullptr, 8);
    std::string fn;
    fn.resize(name_size);
    std::memcpy(fn.data(), datas.data() + sizeof(block), name_size);
    return fn;
  } else {
    return block->header.name;
  }
}

const char* TarEntry::entry_data() { return this->datas.data(); }

size_t TarEntry::entry_size() { return this->datas.size(); }

const char* TarEntry::data() {
  const block* block = (union block*)(this->datas.data());
  bool is_long_name = block->header.typeflag == 'L';
  if (is_long_name) {
    size_t name_size = std::stoul(block->header.size, nullptr, 8);
    auto name_context_size = getNextMultiple(name_size, size_t(512));
    return this->datas.data() + BLOCKSIZE + name_context_size + BLOCKSIZE;
  } else {
    return this->datas.data() + BLOCKSIZE;
  }
}

size_t TarEntry::size() {
  const block* block = (union block*)(this->datas.data());
  bool is_long_name = block->header.typeflag == 'L';
  if (is_long_name) {
    size_t name_size = std::stoul(block->header.size, nullptr, 8);
    auto name_context_size = getNextMultiple(name_size, size_t(512));
    block = (union block*)(this->datas.data() + BLOCKSIZE + name_context_size);
  }
  auto size_ = std::stoul(block->header.size, nullptr, 8);
  return size_;
}

int TarEntry::rename(const std::string& name) {
  block* block = (union block*)(this->datas.data());
  bool is_long_name = block->header.typeflag == 'L';
  bool new_name_is_long_name = name.size() >= sizeof(block->header.name);
  if (!is_long_name && !new_name_is_long_name) {
    // short -> short
    std::memset(block->header.name, 0, sizeof(block->header.name));
    std::memcpy(block->header.name, name.c_str(), name.size());
    // update checksum
    std::memset(block->header.chksum, ' ', 8);
    unsigned int checksum_value = 0;
    for (unsigned int i = 0; i != sizeof(block->buffer); ++i) {
      checksum_value += (uint8_t)block->buffer[i];
    }
    safe_sprintf(block->header.chksum, "%06o", checksum_value);
    return 0;
  } else if (is_long_name && !new_name_is_long_name) {
    // long -> short
    throw "convert long filename to short filename is developing";
  } else {
    throw "long filename not support now";
  }
}

TarEntry::TarEntry(std::vector<char>&& datas) { this->datas = datas; }

} // namespace vaip_tar
