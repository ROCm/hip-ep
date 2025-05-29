/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mem_stream_buffer.hpp"
#include "./mmap_file.hpp"
#include "./tar_entry.hpp"
#include "vaip/export.h"
#include <filesystem>
#include <iostream>
#include <optional>

namespace vaip_core {
class TarFile {
public:
  VAIP_DLL_SPEC static std::unique_ptr<TarFile>
  create(std::unique_ptr<std::iostream>&& stream);

  /**
   * @brief Creates a TarFile instance from the specified file path.
   *
   * This function initializes and returns a unique pointer to a TarFile
   * object, which represents the tar file located at the given file path.
   *
   * this function potentially open a file and create a memory map file, then
   * `mmap` interface method is implemented for open_file_for_read()
   *
   * @param path The file system path to the tar file to be opened or created.
   * @return A unique pointer to the created TarFile instance.
   */
  VAIP_DLL_SPEC static std::unique_ptr<TarFile>
  create(const std::filesystem::path& path);

public:
  TarFile(std::unique_ptr<std::iostream> stream);
  TarFile(std::unique_ptr<MemStream<MemFile>> stream);
  VAIP_DLL_SPEC
  bool has_file(const std::string& filename) const;
  VAIP_DLL_SPEC
  std::vector<std::unique_ptr<TarEntryInputStream>>& entries();
  VAIP_DLL_SPEC
  const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;
  // user must not close this stream.
  // stream->close() is a noop.
  VAIP_DLL_SPEC
  TarEntryInputStream* open_for_read(const std::string& filename);
  // user must close this stream
  // NOTE: there should be only one writer, otherwise return nullptr;
  //
  // there could be many streams already opened for read.
  //
  // if the filename already exists, the orinal entry will be renamed to a wired
  // invisiable name,
  // TODO: check if tar file header support delete flag.
  //
  // after the stream is closed, a new entries is append to the tar stream_.
  //
  // there are 1024 zero bytes at the end of the tar file, which is used to
  // indidcate the end of the tar file. it should be also available after the
  // ostream is closed. so that when stream_ is closed, the tar file is still
  // valid.
  //
  // NOTE: the stream is not thread safe.
  VAIP_DLL_SPEC
  std::unique_ptr<std::ostream> open_for_write(const std::string& filename);

private:
  TarEntryInputStream&
  add_regular_entry(const std::string& path, // path of the entry
                    std::streambuf::pos_type data_begin_pos,
                    std::streambuf::pos_type data_end_pos,
                    std::streambuf::pos_type block_begin_pos,
                    std::streambuf::pos_type block_end_pos);

  TarEntryInputStream*
  add_symlink_entry(const std::string& symlink_name,
                    const std::string& real_path_name,
                    std::streambuf::pos_type block_begin_pos,
                    std::streambuf::pos_type block_end_pos);
  /**
   * @brief Finds the real entry corresponding to the given link name.
   *
   * This function searches for the actual tar entry that the provided link name
   * refers to. It is useful for resolving symbolic links within
   * a tar archive.
   *
   * @param link_name The link name of the entry to resolve.
   * @return A pointer to the TarEntryInputStream representing the real entry,
   *         or nullptr if the entry cannot be found.
   */

  TarEntryInputStream*
  find_real_entry(const std::string& real_path // link name of the entry
  );

  /**
   * @brief Reads a tar entry from the provided stream.
   *
   * This function reads a tar entry from the given stream and returns a
   * TarEntryInputStreamBuffer object representing the entry.
   *
   * @param stream A shared pointer to the input stream to read from.
   * @return A unique pointer to a TarEntryInputStreamBuffer object.
   */
  TarEntryInputStream* read_tar_entry(std::shared_ptr<std::istream> stream);
  TarEntryInputStream* add_entry(const std::string& path, // name of the entry
                                 const std::optional<std::string>& real_path, //
                                 std::streambuf::pos_type data_begin_pos,
                                 std::streambuf::pos_type data_end_pos,
                                 std::streambuf::pos_type block_begin_pos,
                                 std::streambuf::pos_type block_end_pos);

private:
  std::shared_ptr<std::iostream> stream_;
  // it is nullptr if stream_ is not a
  // memory map file. it is stream_.get() if
  // stream_ is a memory map file.
  MemStream<MemFile>* mem_stream_;
  std::vector<std::unique_ptr<TarEntryInputStream>> entries_;
  bool is_writing_{false};
  int padding_count_{0};
  friend class TarEntryOutputStream;
};
} // namespace vaip_core
