/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mem_stream_buffer.hpp"
#include "./mmap_file.hpp"
#include <iostream>
#include <memory>
#include <morphizen/export.h>
#include <optional>
#include <streambuf>
#include <vector>
namespace morphizen {
class TarEntryInputStreamBuffer : public std::streambuf {

public:
  virtual ~TarEntryInputStreamBuffer();

public:
  MORPHIZEN_DLL_SPEC const std::string &path() const;
  MORPHIZEN_DLL_SPEC const std::optional<std::string> &real_path() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type data_begin_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type data_end_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type block_begin_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type block_end_pos() const;
  MORPHIZEN_DLL_SPEC bool is_symlink() const;
  // starting point of real data
  // size of real data
  MORPHIZEN_DLL_SPEC size_t size() const;

public: // make std::unique_ptr happy
  // Assuming `stream` is a valid pointer to a std::istream object,
  // and current read position is at the beginning of a tar header.
  MORPHIZEN_DLL_SPEC explicit TarEntryInputStreamBuffer(
      const std::string &name,                     // name of the entry
      const std::optional<std::string> &real_path, // link name of the entry
      std::streambuf::pos_type data_begin_pos,     // beginning of the data.
      std::streambuf::pos_type data_end_pos,       // end of the data.
      std::streambuf::pos_type block_begin_pos, // beginning of the tar entry.
      std::streambuf::pos_type block_end_pos,   // end of the tar entry.
      std::shared_ptr<std::istream> stream,     //

      std::size_t bufferSize = 4 * 1024 * 1024); // 4 MB: small buffers cause
                                                 // excessive syscalls for
                                                 // multi-GB tar entries
  // Handles reading from FILE*
  virtual int_type underflow() override final;
  // Handles writing to FILE*
  virtual int_type overflow(int_type ch) override;
  // Flushes the output buffer
  // Seek support using fseek
  std::streampos seekoff(std::streamoff offset, std::ios_base::seekdir way,
                         std::ios_base::openmode which) override;
  std::streampos seekpos(std::streampos sp,
                         std::ios_base::openmode which) override final;
  // for logging
  std::string to_string() const;

private:
  const std::string path_;                     // name of the entry
  const std::optional<std::string> real_path_; //

  const pos_type
      data_begin_pos_; // beginning of data (write-once: -1 until first access)
  const pos_type data_end_pos_; // end of data, no padding (write-once: -1 until
                                // first access)
  const pos_type
      block_begin_pos_; // beginning of the tar entry, point to the data.
  const pos_type block_end_pos_; // end of the tar entry
  pos_type buffer_pos_; // current read position where the buffer_ is read from

  std::shared_ptr<std::istream> stream_;
  std::vector<char> buffer_;
  friend class TarFile;

  // other meta info read from the tar header
};

class TarEntryInputStream : public std::istream {

public:
  TarEntryInputStream() = delete;
  // Assuming `stream` is a valid pointer to a std::istream object,
  // and current read position is at the beginning of a tar header.
  MORPHIZEN_DLL_SPEC const std::string &path() const;
  /** @brief  */
  /**
   * @brief Retrieves the real path associated with the tar entry, if available.
   *
   * This function returns an optional string containing the real path. If the
   * real path is not available, the optional will not contain a value, it means
   * that this is a not a symbolic link, but rather a regular file.
   *
   * @return A constant reference to an `std::optional<std::string>`
   * representing the real path of the tar entry.
   */
  MORPHIZEN_DLL_SPEC const std::optional<std::string> &real_path() const;
  // starting point of real data
  // size of real data
  MORPHIZEN_DLL_SPEC size_t size() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type data_begin_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type data_end_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type block_begin_pos() const;
  MORPHIZEN_DLL_SPEC std::streambuf::pos_type block_end_pos() const;
  MORPHIZEN_DLL_SPEC bool is_symlink() const;
  /**
   * @brief Computes the MD5 checksum of the content.
   *
   * This function returns the MD5 hash as a string, which can be used
   * to verify the integrity of the content.
   *
   * @return A string representing the MD5 checksum of the content.
   */
  MORPHIZEN_DLL_SPEC std::string md5(); // MD5 checksum of the content
  std::string to_string() const;        // for logging
  void *mmap();

public: // only for uniqute_ptr
  explicit TarEntryInputStream(std::unique_ptr<TarEntryInputStreamBuffer> buf,
                               MemStream<MemFile> *mem_file);

private:
  std::unique_ptr<TarEntryInputStreamBuffer> buf_;
  MemStream<MemFile> *mem_buf_;
  friend class TarFile;
};
// TarEntryOutputStream is used to write a tar entry to a tar file.
class TarEntryOutputStream : public std::ostream {
public:
  MORPHIZEN_DLL_SPEC static std::unique_ptr<TarEntryOutputStream>
  create(class TarFile &tar_file, const std::string &name);

public:
  TarEntryOutputStream(const std::string &name,            // name of the entry
                       std::streambuf::pos_type begin_pos, // beginning of the
                       class TarFile &tar_file);
  virtual ~TarEntryOutputStream();

private:
  TarEntryOutputStream() = delete;
  static std::streampos
  calculate_tar_append_pos(const TarEntryInputStream &last_entry);
  std::optional<std::string> get_content_check_sum();
  TarEntryInputStream *find_prev_entry_for_md5(const std::string &md5);
  TarEntryInputStream *find_prev_entry_for_path(const std::string &name);
  TarEntryInputStream &add_entry_for_new_data(const std::string &md5);
  void add_symlink_for_existing_entry(const std::string &md5);
  void add_1024_padding();
  static void maybe_add_4k_align(class TarFile &tar_file,
                                 const std::string &name);
  static void add_padding_block_for_4k(class TarFile &tar_file,
                                       const std::string &name, int block_idx);

private:
  const std::string name_; // name of the entry
  const std::streampos
      begin_pos_;          // beginning of the tar entry, point to the data.
  std::streampos end_pos_; // point to  end of the data.
  std::streampos size_;    // size of data
  class TarFile
      &tar_file_; // tar file to write to, use to reset TarFile::is_writing_
};
} // namespace morphizen
