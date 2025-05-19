/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./tar_file.hpp"
#include "./tar_header.hpp"
#include "morphizen/env_config.hpp"
#include <algorithm>
#include <fstream>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= n)
#include <iostream>
namespace vaip_core {

std::unique_ptr<TarFile>
TarFile::create(std::unique_ptr<std::iostream> stream) {
  return std::make_unique<TarFile>(std::move(stream));
}
std::unique_ptr<TarFile> TarFile::create(const std::filesystem::path& path) {
  auto mem_file = std::unique_ptr<MemFile>();
  try {
    mem_file = MemFile::create(path);
    if (mem_file == nullptr) {
      throw std::runtime_error("cannot create mem file for linux");
    }
  } catch (const std::exception& e) {
    MY_LOG(1) << "Failed to create MMapFile object: " << e.what();
    auto stream =
        std::make_unique<std::fstream>(path, std::ios::binary | std::ios::in);
    if (!stream->is_open()) {
      MY_LOG(1) << "Failed to open file: " << path.string();
      return nullptr;
    }
    return std::make_unique<TarFile>(std::move(stream));
  }
  auto base = mem_file->base();
  auto size = mem_file->size();
  auto stream = std::make_unique<MemStream<MemFile>>(
      MemBuffer<MemFile>::create(base, size, std::move(mem_file)));
  return std::make_unique<TarFile>(std::move(stream));
}
TarFile::TarFile(std::unique_ptr<std::iostream> stream)
    : stream_(std::move(stream)), mem_stream_{nullptr} {
  CHECK(stream_->seekg(0, std::ios::beg).good())
      << "Failed to seek to the beginning of the stream";
  // Read the tar header to get the number of entries
  do {
  } while (read_tar_entry(stream_));
}

TarFile::TarFile(std::unique_ptr<MemStream<MemFile>> stream)
    : stream_(nullptr), mem_stream_{nullptr} {
  mem_stream_ = stream.get();
  stream_ = std::move(stream);
  CHECK(stream_->seekg(0, std::ios::beg).good())
      << "Failed to seek to the beginning of the stream";
  // Read the tar header to get the number of entries
  do {
  } while (read_tar_entry(stream_));
}
bool TarFile::has_file(const std::string& filename) const {
  for (auto& entry : entries()) {
    if (entry->path() == filename) {
      MY_LOG(1) << " has_file: Found entry=" << entry->to_string() //
                << " stream_pos=" << entry->tellg()                //
          ;
      return true;
    }
  }
  return false;
}

std::vector<std::unique_ptr<TarEntryInputStream>>& TarFile::entries() {
  return entries_;
}
const std::vector<std::unique_ptr<TarEntryInputStream>>&
TarFile::entries() const {
  return entries_;
}
TarEntryInputStream* TarFile::open_for_read(const std::string& filename) {
  MY_LOG(1) << " open_for_read: search for file \"" << filename << "\"";
  for (auto& entry : entries_) {
    if (entry->path() == filename) {
      if (entry->data_begin_pos() == std::streampos(-1)) {
        auto real_entry = find_real_entry(filename);
        if (!real_entry) {
          MY_LOG(1) << " open_for_read: entry \"" << entry->to_string()
                    << "\" not found in the tar file";
          return nullptr;
        }
        const_cast<std::streampos&>(entry->buf_->data_begin_pos_) =
            real_entry->data_begin_pos(); // set the data begin pos
        const_cast<std::streampos&>(entry->buf_->data_end_pos_) =
            real_entry->data_end_pos(); // set the data begin pos
        const_cast<std::streampos&>(entry->buf_->buffer_pos_) =
            real_entry->data_begin_pos(); // set the data begin pos
        return entry.get();
      }
      entry->clear();
      if (!entry->seekg(0, std::ios::beg).good()) {
        MY_LOG(1) << "Failed to seek to the beginning of the entry stream"
                  << " entry name=" << entry->path()
                  << " entry size=" << entry->size();
        return nullptr;
      }
      MY_LOG(1) << " open_for_read: Found entry=" << entry->to_string() //
                << " stream_pos=" << entry->tellg()                     //
          ;
      return entry.get();
    }
  }
  return nullptr;
}

std::unique_ptr<std::ostream>
TarFile::open_for_write(const std::string& filename) {
  if (mem_stream_) {
    // mem_stream_ is not nullptr, it means tar file is created in memory, it is
    // readonly, we cannot expand the size of memroy dynamically
    return nullptr;
  }
  return TarEntryOutputStream::create(*this, filename);
}
TarEntryInputStream&
TarFile::add_regular_entry(const std::string& path, // name of the entry
                           std::streambuf::pos_type data_begin_pos,
                           std::streambuf::pos_type data_end_pos,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // erase the old entry if found
  auto it = std::remove_if(
      entries_.begin(), entries_.end(), [&path](const auto& entry) {
        auto ret = entry->path() == path;
        if (ret) {
          MY_LOG(1)
              << " add_symlink_entry: duplicated entry found, remove old entry "
              << entry->to_string() //
              ;
        }
        return ret;
      });
  if (it != entries_.end()) {
    entries_.erase(it, entries_.end());
  }

  auto ret = add_entry(path, std::nullopt, data_begin_pos, data_end_pos,
                       block_begin_pos, block_end_pos);

  MY_LOG(1) << " add entry: \"" << ret->to_string() << " " << entries_.size()
            << " in total";
  return *ret;
}

TarEntryInputStream*
TarFile::find_real_entry(const std::string& real_path // link name of the entry
) {
  auto it = std::find_if(
      entries_.rbegin(), entries_.rend(),
      [&real_path](const auto& entry) { return entry->path() == real_path; });
  MY_LOG(1) << " find_real_entry: search for symlink \"" << real_path << "\"";
  if (it != entries_.rend()) {
    if ((*it)->real_path()) {
      MY_LOG(1) << " find_real_entry: Found entry \"" << (*it)->path()
                << "\", but it is a symlink to \""
                << " link_name=" << (*it)->real_path().value()
                << " search recursively for the real entry" //
          ;
      return find_real_entry((*it)->real_path().value());
    } else {
      MY_LOG(1) << " find_real_entry: Found entry \"" << (*it)->path()
                << "\""                                     //
                << " size=" << (*it)->size()                //
                << " begin_pos=" << (*it)->data_begin_pos() //
                << " end_pos=" << (*it)->data_end_pos()     //
          ;                                                 //
      return (*it).get();
    }
  } else {
    MY_LOG(1) << " find_real_entry: entry \"" << real_path << "\" not found";
  }
  return nullptr;
}

TarEntryInputStream*
TarFile::add_symlink_entry(const std::string& symlink_name,
                           const std::string& real_path_name,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // erase the old entry if found
  auto it = std::remove_if(
      entries_.begin(), entries_.end(), [&symlink_name](const auto& entry) {
        auto ret = entry->path() == symlink_name;
        if (ret) {
          MY_LOG(1)
              << " add_symlink_entry: duplicated entry found, remove old entry "
              << entry->to_string() //
              ;
        }
        return ret;
      });
  if (it != entries_.end()) {

    entries_.erase(it, entries_.end());
  }
  TarEntryInputStream* ret = nullptr;
  auto real_entry = find_real_entry(real_path_name);
  if (real_entry) {
    ret = add_entry(symlink_name, real_path_name, real_entry->data_begin_pos(),
                    real_entry->data_end_pos(), block_begin_pos, block_end_pos);
    MY_LOG(1) << " add_symlink_entry: found real entry:"
              << real_entry->to_string()
              << " symlink entry:" << ret->to_string();
  } else {
    ret = add_entry(symlink_name, real_path_name, std::streampos(-1),
                    std::streampos(-1), block_begin_pos, block_end_pos);
    MY_LOG(1) << " add_symlink_entry: cannot find real entry:"
              << " symlink entry:" << ret->to_string();
  }
  return ret;
}
static bool is_padding_header(const TarHeader& header) {
  auto is_special_name = [](const std::string& name) {
    static const char prefix[] = "_data/padding_";
    return std::memcmp(name.data(), prefix,
                       std::min(sizeof(prefix) - 1u, name.size())) == 0;
  };
  auto ret = true;
  ret = ret && !header.is_symlink();
  ret = ret && header.size() == 0u;
  ret = ret && is_special_name(header.path());
  return ret;
}
TarEntryInputStream*
TarFile::read_tar_entry(std::shared_ptr<std::istream> stream) {
  auto tar_header = TarHeader::read_header(*stream);
  while (tar_header && is_padding_header(*tar_header)) {
    tar_header = TarHeader::read_header(*stream);
  }
  if (!tar_header) {
    MY_LOG(1) << " might read end of file";
    stream_->clear(); // clear the eof flag
    return nullptr;
  }
  TarEntryInputStream* ret = nullptr;
  if (tar_header->real_path()) {
    MY_LOG(1) << " read symlink entry: " << tar_header->to_string();
    ret = add_symlink_entry(tar_header->path(),              //
                            tar_header->real_path().value(), //
                            tar_header->block_begin_pos(),   //
                            tar_header->block_end_pos()      //
    );
  } else {
    MY_LOG(1) << " read entry \"" << tar_header->to_string();
    ;
    ret = &add_regular_entry(tar_header->path(),            //
                             tar_header->data_begin_pos(),  //
                             tar_header->data_end_pos(),    //
                             tar_header->block_begin_pos(), //
                             tar_header->block_end_pos()    //
    );
  }
  return ret;
}
TarEntryInputStream*
TarFile::add_entry(const std::string& path, // name of the entry
                   const std::optional<std::string>& real_path,
                   std::streambuf::pos_type data_begin_pos,
                   std::streambuf::pos_type data_end_pos,
                   std::streambuf::pos_type block_begin_pos,
                   std::streambuf::pos_type block_end_pos) {

  auto entry = std::make_unique<TarEntryInputStream>(
      std::make_unique<TarEntryInputStreamBuffer>(path,
                                                  real_path,       //
                                                  data_begin_pos,  //
                                                  data_end_pos,    //
                                                  block_begin_pos, //
                                                  block_end_pos,   //
                                                  stream_),
      mem_stream_);
  auto ret = entry.get();
  MY_LOG(1) << " add_entry \"" << ret->to_string();
  entries_.emplace_back(std::move(entry));
  return ret;
}
} // namespace vaip_core
