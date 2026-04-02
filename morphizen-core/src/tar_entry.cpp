/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include "./tar_entry.hpp"
#include "./tar_file.hpp"
#include "md5.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "tar_header.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= n)

namespace morphizen {
static constexpr auto BLOCKSIZE = 512;

TarEntryInputStreamBuffer::TarEntryInputStreamBuffer(
    const std::string& name,                     //
    const std::optional<std::string>& real_path, //
    std::streambuf::pos_type data_begin_pos,     // beginning of the data.
    std::streambuf::pos_type data_end_pos,       // end of the data.
    std::streambuf::pos_type block_begin_pos,    // beginning of the tar entry.
    std::streambuf::pos_type block_end_pos,      // end of the tar entry.
    std::shared_ptr<std::istream> stream,        //
    std::size_t bufferSize)
    : path_{name},                               //
      real_path_{real_path},                     //
      data_begin_pos_{data_begin_pos},           //
      data_end_pos_{data_end_pos},               //
      block_begin_pos_{block_begin_pos},         //
      block_end_pos_{block_end_pos},             //
      buffer_pos_{block_begin_pos},              //
      stream_{stream},                           //
      buffer_(std::min(bufferSize, static_cast<size_t>(std::max<std::streamoff>(
                                       1, data_end_pos - data_begin_pos)))) {
  setg(buffer_.data(), buffer_.data(), buffer_.data());
  // does not support writing.
  setp(nullptr, nullptr); // Set write buffer
  // Set read buffer
  // set the end_pos_ according to the TAR header
}

TarEntryInputStreamBuffer::~TarEntryInputStreamBuffer() {}

const std::string& TarEntryInputStreamBuffer::path() const { return path_; }
const std::optional<std::string>& TarEntryInputStreamBuffer::real_path() const {
  return real_path_;
}
std::streambuf::pos_type TarEntryInputStreamBuffer::data_begin_pos() const {
  return data_begin_pos_;
}
std::streambuf::pos_type TarEntryInputStreamBuffer::data_end_pos() const {
  return data_end_pos_;
}
std::streambuf::pos_type TarEntryInputStreamBuffer::block_begin_pos() const {
  return block_begin_pos_;
}
std::streambuf::pos_type TarEntryInputStreamBuffer::block_end_pos() const {
  return block_end_pos_;
}
bool TarEntryInputStreamBuffer::is_symlink() const {
  return real_path_.has_value();
}
std::string TarEntryInputStreamBuffer::to_string() const {
  std::ostringstream ret;
  ret << "TarEntryInputStreamBuffer{"
      << "\"" << path_ << "\"" //
      ;
  if (real_path_) {
    ret << " -> \"" << real_path_.value() << "\"";
  }
  ret << " size=" << size()                                                 //
      << " block_pos=(" << block_begin_pos_ << "," << block_end_pos_ << ")" //
      << " data_pos=(" << data_begin_pos_ << "," << data_end_pos_ << ")"    //
      << " buffer_size=" << buffer_.size()                                  //
      << " buffer_pos=" << buffer_pos_                                      //
      << " stream_pos=" << stream_->tellg()                                 //
      << "} ";

  return ret.str();
}
size_t TarEntryInputStreamBuffer::size() const {
  auto size = data_end_pos_ - data_begin_pos_;
  CHECK(size >= 0) << "size=" << size << " data_begin_pos_=" << data_begin_pos_
                   << " data_end_pos_=" << data_end_pos_;
  return (size_t)size;
}
std::streambuf::int_type TarEntryInputStreamBuffer::underflow() {
  if (buffer_pos_ >= data_end_pos_) {
    MY_LOG(1) << " EOF: buffer_pos_ " << buffer_pos_ << " " //
              << "data_end_pos_ " << data_end_pos_ << " "   //
        ;
    return traits_type::eof();
  }
  CHECK(stream_->seekg(buffer_pos_).good())
      << "seekg failed. buffer_pos_=" << buffer_pos_;
  auto leftSize = data_end_pos_ - buffer_pos_;
  CHECK(leftSize >= 0) << "leftSize=" << leftSize
                       << " buffer_pos_=" << buffer_pos_
                       << " end_pos_=" << data_end_pos_;
  auto bufferSize = buffer_.size();
  auto readSize = std::min(bufferSize, (size_t)leftSize);
  CHECK(stream_->read(buffer_.data(), readSize).good())
      << "read failed. buffer_pos_=" << buffer_pos_;
  auto bytesRead = stream_->gcount();
  if (bytesRead == 0) {
    MY_LOG(1) << " EOF: buffer_pos_ " << buffer_pos_ << " " //
              << "data_end_pos_ " << data_end_pos_ << " "   //
              << "stream_pos " << stream_->tellg() << " "   //
              << "bytesRead " << bytesRead << " "           //
        ;
    return traits_type::eof();
  }
  buffer_pos_ += std::streampos(bytesRead);
  setg(buffer_.data(), buffer_.data(), buffer_.data() + bytesRead);
  MY_LOG(3) << " read more data from tar file. etry=" << to_string();
  return traits_type::to_int_type(*gptr());
}
std::streambuf::int_type TarEntryInputStreamBuffer::overflow(int_type ch) {
  CHECK(false) << " do not support writing"
               << " ch=" << ch;
  return traits_type::eof();
}
std::streampos
TarEntryInputStreamBuffer::seekoff(std::streamoff offset,
                                   std::ios_base::seekdir way,
                                   std::ios_base::openmode which) {
  if (which & std::ios_base::in) {
    if (way == std::ios_base::beg) {
      buffer_pos_ = data_begin_pos_ + offset;
    } else if (way == std::ios_base::cur) {
      buffer_pos_ += gptr() - eback() + offset;
    } else if (way == std::ios_base::end) {
      buffer_pos_ = data_end_pos_ + offset;
    }
    setg(buffer_.data(), buffer_.data(), buffer_.data());
  } else if (which & std::ios_base::out) {
    CHECK(false) << " do not support writing";
  }
  return buffer_pos_ - data_begin_pos_;
}

std::streampos
TarEntryInputStreamBuffer::seekpos(std::streampos sp,
                                   std::ios_base::openmode which) {
  // optionally call seekoff() here to centralize logic
  return seekoff(sp, std::ios_base::beg, which);
}

TarEntryInputStream::TarEntryInputStream(
    std::unique_ptr<TarEntryInputStreamBuffer> buf, MemStream<MemFile>* mem_buf)
    : std::istream(buf.get()), buf_{nullptr}, mem_buf_{mem_buf} {
  buf_ = std::move(buf);
}

const std::string& TarEntryInputStream::path() const { return buf_->path(); }
const std::optional<std::string>& TarEntryInputStream::real_path() const {
  return buf_->real_path();
}
size_t TarEntryInputStream::size() const { return buf_->size(); }

std::streambuf::pos_type TarEntryInputStream::data_begin_pos() const {
  return buf_->data_begin_pos();
}
std::streambuf::pos_type TarEntryInputStream::data_end_pos() const {
  return buf_->data_end_pos();
}
std::streambuf::pos_type TarEntryInputStream::block_begin_pos() const {
  return buf_->block_begin_pos();
}
std::streambuf::pos_type TarEntryInputStream::block_end_pos() const {
  return buf_->block_end_pos();
}
bool TarEntryInputStream::is_symlink() const { return buf_->is_symlink(); }
std::string TarEntryInputStream::md5() {
  // calculate md5 sum of the stream
  clear();
  this->seekg(0);
  if (!this->good()) {
    MY_LOG(1) << "seekg failed. begin_pos=" << 0 << "size=" << size() //
              << " stream " << this->tellg() << "stream.fail() " << this->fail()
              << " "                                                  //
              << "stream.bad() " << this->bad() << " "                //
        ;
    return "";
  }
  auto ret = MD5();
  auto buffer = std::vector<char>(4096);
  auto total_size = size();
  while (total_size > 0) {
    auto read_size = std::min(total_size, buffer.size());
    this->read(buffer.data(), read_size);
    if (!this->good()) {
      MY_LOG(3) << "read failed. size=" << size();
      return "";
    }
    ret.add(buffer.data(), read_size);
    total_size -= read_size;
  }
  this->clear();
  return ret.getHash();
}

void* TarEntryInputStream::mmap() {
  if (mem_buf_) {
    return (void*)mem_buf_->offset(buf_->data_begin_pos());
  }
  return nullptr;
}
std::string TarEntryInputStream::to_string() const {
  return std::string("TarEntryInputStream{buf=") + buf_->to_string() + "}";
}
TarEntryOutputStream::TarEntryOutputStream(
    const std::string& name,            // name of the entry
    std::streambuf::pos_type begin_pos, // beginning of the
    class TarFile& tar_file)
    : std::ostream(tar_file.stream_->rdbuf()), name_{name},
      begin_pos_{begin_pos}, tar_file_{tar_file} {
  MY_LOG(2) << " assume the write position is set by "
               "TarEntryOutputStream::create."    //
            << " begin_pos " << begin_pos_ << " " //
            << " stream_pos: " << tellp() << " ";
}
static std::string calculate_md5(std::istream& str, std::streampos begin_pos,
                                 std::streamsize size) {
  // calculate md5 sum of the stream
  str.seekg(begin_pos);
  CHECK(str.good()) << "seekg failed. begin_pos=" << begin_pos
                    << "size=" << size                     //
                    << " stream " << str.tellg() << "stream.fail() "
                    << str.fail() << " "                   //
                    << "stream.bad() " << str.bad() << " " //
      ;
  auto md5 = MD5();
  auto buffer = std::vector<char>(4096);
  while (size > 0) {
    auto read_size = std::min(size, (std::streamsize)buffer.size());
    str.read(buffer.data(), read_size);
    if (!str.good()) {
      MY_LOG(3) << "read failed. size=" << size;
      return "";
    }
    md5.add(buffer.data(), read_size);
    size -= read_size;
  }
  return md5.getHash();
}

std::optional<std::string> TarEntryOutputStream::get_content_check_sum() {
  auto current_pos = this->tellp();
  auto end_pos = current_pos;
  this->seekp(begin_pos_);
  if (!this->good()) {
    MY_LOG(1) << "cannot rewind to the origina position."
              << "begin_pos_ " << begin_pos_ << " "   //
              << "current_pos " << current_pos << " " //
              << " stream_pos: " << tellp() << " "    //
        ;
    return std::nullopt;
  }
  auto size = current_pos - begin_pos_;
  MY_LOG(1) << " " << size << " bytes were written to file"
            << " \"" << name_ << "\""
            << " from " << begin_pos_   //
            << " to " << end_pos << "." //
            << " start to submit a new tar entry.";
  // write padding
  // tar file is block aligned, so we need to pad the file to the next block
  auto padding_size = round_up_to_block_size(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    this->seekp(current_pos);
    this->write(padding.data(), padding.size());
    end_pos = this->tellp();
    if (!this->good()) {
      CHECK(false) << "write padding failed. size=" << size
                   << " current_pos=" << current_pos
                   << " begin_pos_=" << begin_pos_;
    } else {
      MY_LOG(2) << " write " << padding.size() << " bytes for padding" //
                << " from " << current_pos                             //
                << " to " << end_pos                                   //
                << ", stream pos=" << tellp()                          //
          ;
    }
  }
  // calculate md5 sum of written data.
  auto ret = calculate_md5(*this->tar_file_.stream_.get(), begin_pos_, size);
  this->seekp(end_pos);
  MY_LOG(1) << " md5 of file"                                      //
            << " \"" << name_ << "\" (" << size << " bytes)"       //
            << " is " << ret                                       //
            << " from " << begin_pos_                              //
            << " to " << current_pos << ", padding to " << end_pos //
            << " stream_pos=" << tellp()                           //
      ;
  end_pos_ = end_pos;
  size_ = size;
  return ret;
}

TarEntryInputStream*
TarEntryOutputStream::find_prev_entry_for_md5(const std::string& md5) {
  auto data_file_name = std::string("_data/") + md5;

  for (auto& entry : tar_file_.entries_) {
    if (entry->path() == data_file_name) {
      MY_LOG(1) << " duplicated data found for entry "  //
                << entry->to_string()                   //
                << " data_file_name=" << data_file_name //
          ;
      return entry.get();
    }
  }
  return nullptr;
}

// FIXME , the param "name" is not used
TarEntryInputStream*
TarEntryOutputStream::find_prev_entry_for_path(const std::string& /*name*/) {
  for (auto& entry : tar_file_.entries_) {
    if (entry->path() == name_) {
      MY_LOG(1) << " duplicated file found for entry " //
                << entry->to_string()                  //
          ;
      return entry.get();
    }
  }
  return nullptr;
}
TarEntryInputStream&
TarEntryOutputStream::add_entry_for_new_data(const std::string& md5) {
  auto data_file_name = std::string("_data/") + md5;
  MY_LOG(1) << " " << data_file_name << " does not exists, create a new entry";
  seekp(begin_pos_ - static_cast<std::streamoff>(512));
  if (!good()) {
    CHECK(false) << "seek to write header failed. " //
                 << "begin_pos_=" << begin_pos_     //
                 << " size=" << size_               //
                 << " end_pos=" << end_pos_         //
                 << " stream pos=" << tellp()       //
        ;
  }
  auto header = TarHeader(data_file_name, size_);
  header.write_header(*this);
  if (!this->good()) {
    MY_LOG(1) << "write header failed. name=" << name_;
  } else {
    MY_LOG(1) << " write header for " << data_file_name << " size=" << size_
              << " OK. header=" << header.to_string();
  }
  auto& ret = tar_file_.add_regular_entry(data_file_name,           //
                                          header.data_begin_pos(),  //
                                          header.data_end_pos(),    //
                                          header.block_begin_pos(), //
                                          header.block_end_pos()    //
  );
  // go back to the original position
  this->seekp(end_pos_);
  auto sym_header = TarHeader(name_, 0);
  sym_header.set_link_name(data_file_name);
  sym_header.write_header(*this);
  if (!this->good()) {
    MY_LOG(1) << "write symbol header failed. name=" << name_;
  } else {
    MY_LOG(1) << " write header for symlink."         //
              << " header=" << sym_header.to_string() //
              << " stream_pos=" << tellp();
  }
  tar_file_.add_symlink_entry(name_, data_file_name,
                              sym_header.block_begin_pos(),
                              sym_header.block_end_pos());

  return ret;
}

void TarEntryOutputStream::add_symlink_for_existing_entry(
    const std::string& md5) {
  auto data_file_name = std::string("_data/") + md5;
  // ignore the written data, probably garbage after 1024 bytes
  seekp(begin_pos_ - static_cast<std::streampos>(512));
  if (!good()) {
    MY_LOG(1) << "seek to write header failed. " //
              << "begin_pos_=" << begin_pos_     //
              << " size=" << size_               //
              << " end_pos=" << end_pos_         //
              << " stream pos=" << tellp()       //
        ;
    return;
  }
  auto sym_header = TarHeader(name_, 0);
  sym_header.set_link_name(data_file_name);
  sym_header.write_header(*this);
  if (!this->good()) {
    MY_LOG(1) << "write symbol header failed. name=" << name_;
  } else {
    MY_LOG(1) << " write header for symlink " << name_
              << ", to=" << data_file_name << " OK." //
              << "begin_pos_=" << begin_pos_         //
              << " size=" << size_                   //
              << " end_pos=" << end_pos_             //
              << " stream_pos=" << tellp();
  }
  tar_file_.add_symlink_entry(name_, data_file_name,
                              sym_header.block_begin_pos(),
                              sym_header.block_end_pos());
}

void TarEntryOutputStream::add_1024_padding() {
  // write 1024 bytes of zeros at the end of the file
  std::string zeros(1024, '\0');
  this->write(zeros.data(), zeros.size());
  if (!this->good()) {
    MY_LOG(1) << "write zeros failed. name=" << name_;
  }
  this->flush();
}
void TarEntryOutputStream::maybe_add_4k_align(TarFile& tar_file,
                                              const std::string& name) {
  static const auto const_512 = std::streamoff(512);
  static const auto const_4k = std::streamoff(4096);
  auto current_pos = tar_file.stream_->tellp();
  tar_file.stream_->seekp(current_pos -
                          const_512); // rewind to the block header
  if (!tar_file.stream_->good()) {
    MY_LOG(1) << "seekp failed. current_pos=" << current_pos << " name=" << name
              << " stream_pos=" << tar_file.stream_->tellp();
    CHECK(false);
  }
  // the first block might be overwritten if it happens to be 4k aligned
  auto block_idx = 0;
  add_padding_block_for_4k(tar_file, name, block_idx++);
  auto next_data_begin = tar_file.stream_->tellp();
  while (next_data_begin % const_4k != 0) {
    add_padding_block_for_4k(tar_file, name, block_idx++);
    next_data_begin = tar_file.stream_->tellp();
  }
  return;
}
void TarEntryOutputStream::add_padding_block_for_4k(TarFile& tar_file,
                                                    const std::string& name,
                                                    int /*block_idx*/) {
  auto short_name = name.size() < 50u ? name : name.substr(0, 50);
  auto data_file_name = std::string("_data/") + "padding_" +
                        std::to_string(tar_file.padding_count_++);
  auto sym_header = TarHeader(data_file_name, 0);
  sym_header.write_header(*tar_file.stream_);
}

TarEntryOutputStream::~TarEntryOutputStream() {
  tar_file_.is_writing_ = false;
  auto md5 = get_content_check_sum();
  if (!md5) {
    MY_LOG(1) << "cannot calculate md5 for file \"" << name_ << "\"";
    return;
  }
  auto prev_entry_for_md5 = find_prev_entry_for_md5(*md5);
  auto prev_entry_for_path = find_prev_entry_for_path(name_);
  auto same_data_exists = prev_entry_for_md5 != nullptr;
  auto same_path_exists = prev_entry_for_path != nullptr;
  if (!same_data_exists && !same_path_exists) {
    // this is the very first case for a new file and new data.
    // TAR file is designed to be append only, so we need to add a new entry for
    // the later entry wins if the file name is the same.
    add_entry_for_new_data(md5.value());
    add_1024_padding();
  } else if (same_data_exists && !same_path_exists) {
    // this is the second case for shared data, but different file name.
    add_symlink_for_existing_entry(md5.value());
    add_1024_padding();
  } else if (same_data_exists && same_path_exists) {
    // this is the third case for shared data and same file name.
    // we do nothing more than write 1024 bytes of zeros at the end.
    MY_LOG(1) << " same data and same file name exists. do nothing more."
              << " name=" << name_ << " md5=" << md5.value() //
              << " prev_entry_for_md5=" << prev_entry_for_md5->to_string()
              << " prev_entry_for_path=" << prev_entry_for_path->to_string();
    CHECK(seekp(begin_pos_ - std::streampos(512)).good());
    add_1024_padding();
  } else if (!same_data_exists && same_path_exists) {
    // this is the fourth case for different data and same file name.
    // we need to rename the old entry to a new name.
    // TAR is append only, append the new entry to the end of the file.
    // this is as same as the first case.
    add_entry_for_new_data(md5.value());
    add_1024_padding();
  } else {
    CHECK(false) << "never goes here. this is a bug. "       //
                 << " same_data_exists=" << same_data_exists //
                 << " same_path_exists=" << same_path_exists //
        ;
  }
  tar_file_.stream_->flush();
}

std::streampos TarEntryOutputStream::calculate_tar_append_pos(
    const TarEntryInputStream& last_entry) {
  // calculate the tar append position
  // the tar file end is 1024 bytes, so we need to add 512 bytes for the
  // header and padding

  // when the last entry is a symlink, we need to use the block_end_pos,
  // because data_end_pos is for the real data, and symbolic link does not
  // have real data. the block_end_pos is the end of the tar entry, which is
  // the end of the block.
  auto end_pos = last_entry.is_symlink() ? last_entry.block_end_pos()
                                         : last_entry.data_end_pos();
  std::streampos padding_size = 0;
  if (end_pos % BLOCKSIZE != 0) {
    padding_size = BLOCKSIZE - (end_pos % BLOCKSIZE);
  }
  auto append_pos =
      end_pos + std::streampos(padding_size) + std::streampos(BLOCKSIZE);
  MY_LOG(1) << " caculate the writing position for a new file. the last "
            << " entry=" << last_entry.to_string()
            << ". reserve 512 bytes for header, start to write at " //
            << append_pos                                           //
      ;
  return append_pos;
}
std::unique_ptr<TarEntryOutputStream>
TarEntryOutputStream::create(class TarFile& tar_file, const std::string& name) {
  if (tar_file.is_writing_) {
    MY_LOG(1) << "TarFile is already in writing mode";
    return nullptr;
  }
  tar_file.is_writing_ = true;
  // we assume that the tar file stream always has a valid position, and a
  // valid 1024 zeros at end, so that seekp is usually good.

  // begin_pos is the start to data, so that begin_pos - 512 is reverved for
  // the tar block. we always use md5 checksum as a unique file name so that
  // it won't be larger than 100 bytes. it is safe to only reserve 1 block,
  // because it won't need extra block for long file names.
  std::streampos begin_pos = 0;
  if (tar_file.entries_.empty()) {
    // write 1024 zeros at the end of the file
    std::string zeros(1024, '\0');
    tar_file.stream_->write(zeros.data(), zeros.size());
    if (tar_file.stream_->fail()) {
      MY_LOG(1) << "write zeros failed."                              //
                << "begin_pos " << begin_pos << " "                   //
                << "stream ppos " << tar_file.stream_->tellp() << " " //
                << "stream gpos " << tar_file.stream_->tellg() << " " //
                << "good " << tar_file.stream_->good() << " "         //
                << "bad " << tar_file.stream_->bad() << " "           //
                << "fail " << tar_file.stream_->fail() << " "         //
          ;
      return nullptr;
    }
    begin_pos = std::streampos(BLOCKSIZE);
  } else {
    begin_pos = calculate_tar_append_pos(*tar_file.entries_.back());
  }
  tar_file.stream_->seekp(begin_pos);
  if (!tar_file.stream_->good()) {
    MY_LOG(1) << "seek to append position failed."                 //
              << "begin_pos " << begin_pos << " "                  //
              << "stream pos " << tar_file.stream_->tellp() << " " //
        ;
    return nullptr;
  }
  // write 1024 zeros at the end of the file
  std::string zeros(1024, '\0');
  tar_file.stream_->write(zeros.data(), zeros.size());
  if (!tar_file.stream_->good()) {
    MY_LOG(1) << "write zeros failed."                             //
              << "begin_pos " << begin_pos << " "                  //
              << "stream pos " << tar_file.stream_->tellp() << " " //
        ;
    return nullptr;
  }
  tar_file.stream_->seekp(begin_pos);
  if (!tar_file.stream_->good()) {
    MY_LOG(1) << "seek to append position failed."                 //
              << "begin_pos " << begin_pos << " "                  //
              << "stream pos " << tar_file.stream_->tellp() << " " //
        ;
    return nullptr;
  }
  maybe_add_4k_align(tar_file, name);
  MY_LOG(1) << " TarEntryOutputStream::create: start to writing to file \""
            << name << "\" from " << tar_file.stream_->tellp()
            << " begin_pos=" << begin_pos;
  return std::make_unique<TarEntryOutputStream>(name, tar_file.stream_->tellp(),
                                                tar_file);
}

} // namespace morphizen
