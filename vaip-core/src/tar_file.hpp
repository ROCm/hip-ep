#pragma once
#include "./tar_entry.hpp"
#include "vaip/export.h"
#include <iostream>
#include <optional>

namespace vaip_core {
class TarFile {
public:
  VAIP_DLL_SPEC static std::unique_ptr<TarFile>
  create(std::unique_ptr<std::iostream> stream);

public:
  TarFile(std::unique_ptr<std::iostream> stream);
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
  TarEntryInputStream& add_entry(const std::string& path, // path of the entry
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

private:
  std::shared_ptr<std::iostream> stream_;
  std::vector<std::unique_ptr<TarEntryInputStream>> entries_;
  bool is_writing_{false};
  friend class TarEntryOutputStream;
};
} // namespace vaip_core
