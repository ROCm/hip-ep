/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mem_stream_buffer.hpp"
#include "./mmap_file.hpp"
#include "./tar_entry.hpp"
#include "morphizen/dll_safe.h"
#include "morphizen/export.h"
#include <filesystem>
#include <iostream>
#include <optional>
namespace morphizen {

class TarFile {
public:
  MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile>
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
  static std::unique_ptr<TarFile>
  create_from_path(const std::filesystem::path& path, bool enable_mmap = true);

  /**
   * @brief Creates a TarFile instance from a temporary file.
   *
   * Creates an empty tar file using tmpfile() (or tmpfile_with_posix_delete()
   * on Windows). Falls back to in-memory stringstream if tmpfile creation
   * fails.
   *
   * @return A unique pointer to the created TarFile instance.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile> create_from_tmpfile();
  /**
   * @brief Creates a TarFile instance from a vector buffer.
   *
   * Takes ownership of the buffer and wraps it in a TarFile.
   *
   * @param buffer A vector of characters containing the tar file data.
   * @return A unique pointer to the created TarFile instance.
   */
  static std::unique_ptr<TarFile>
  create_from_buffer(std::vector<char>&& buffer);

  /**
   * @brief Creates a TarFile instance from a string buffer.
   *
   * Takes ownership of the buffer and wraps it in a TarFile. Optionally writes
   * to tmpfile and uses memory mapping for better performance.
   *
   * @param buffer A string containing the tar file data.
   * @param enable_mmap If true, attempt to use memory mapping for reads
   * @return A unique pointer to the created TarFile instance.
   */
  static std::unique_ptr<TarFile> create_from_buffer(std::string&& buffer,
                                                     bool enable_mmap = true);

private:
  struct PrivateTag {
  }; // PrivateTag pattern - see docs/technical/privatetag-factory-pattern.md
public:
  TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream);

  MORPHIZEN_DLL_SPEC
  bool has_file(const std::string& filename) const;
  MORPHIZEN_DLL_SPEC
  const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;
  // user must not close this stream.
  // stream->close() is a noop.
  MORPHIZEN_DLL_SPEC
  TarEntryInputStream* open_for_read(const std::string& filename);
  // user must close this stream
  // NOTE: there should be only one writer, otherwise return nullptr;
  //
  // there could be many streams already opened for read.
  //
  // If the filename already exists, a new entry is appended to the tar file.
  // When reading, the last entry with a given name takes precedence (standard
  // TAR semantics). The tar file is append-only; existing entries are never
  // modified.
  //
  // after the stream is closed, a new entry is appended to the tar stream_.
  //
  // there are 1024 zero bytes at the end of the tar file, which is used to
  // indicate the end of the tar file. it should be also available after the
  // ostream is closed. so that when stream_ is closed, the tar file is still
  // valid.
  //
  // NOTE: the stream is not thread safe.
  MORPHIZEN_DLL_SPEC
  std::unique_ptr<std::ostream> open_for_write(const std::string& filename);
  /**
   * @brief Returns the current size of the object.
   *
   * This function provides the current size, typically representing
   * the number of bytes for the tar file.
   *
   * @code
   *     auto size = tar_file.current_size();
   *     auto buf = std::vector<char>(size);
   *     tar_file.dump_to(buf.data(), buffer.size())
   * @return The current size as a value of type size_t.
   */
  MORPHIZEN_DLL_SPEC
  size_t current_size() const;
  /**
   * @brief Dumps the contents of the object to the provided buffer.
   *
   * Copies the tar file content into the specified buffer up to the given size.
   *
   * @param data Pointer to the destination buffer where the data will be
   * written.
   * @param size The maximum number of bytes to write to the buffer.
   * @return return true if sucess, false otherise.
   * value on error.
   * @note if the size is less than the current size of the tar file, it results
   * in a corrupted tar file
   */
  bool dump_to(char* data, size_t size) const;

private:
  /// Adds a regular file entry to the tar archive.
  /// Removes any existing entry with the same path (TAR last-wins semantics).
  /// Regular entries store actual file data at specified positions in the tar
  /// stream.
  /// @param path - Entry path in the archive
  /// @param data_begin_pos - Start position of file data in tar stream
  /// @param data_end_pos - End position of file data in tar stream
  /// @param block_begin_pos - Start position of tar block (including header)
  /// @param block_end_pos - End position of tar block (including padding)
  /// @return Reference to the created entry input stream
  TarEntryInputStream&
  add_regular_entry(const std::string& path, // path of the entry
                    std::streambuf::pos_type data_begin_pos,
                    std::streambuf::pos_type data_end_pos,
                    std::streambuf::pos_type block_begin_pos,
                    std::streambuf::pos_type block_end_pos);

  /// Adds a symlink entry to the tar archive.
  /// Removes any existing entry with the same path (TAR last-wins semantics).
  /// Symlinks point to another entry; this method resolves the target via
  /// find_real_entry(). If the target is not found, creates a symlink with
  /// invalid positions (-1) for lazy resolution.
  /// @param symlink_name - Symlink path in the archive
  /// @param real_path_name - Target path that the symlink points to
  /// @param block_begin_pos - Start position of tar block (including header)
  /// @param block_end_pos - End position of tar block (including padding)
  /// @return Pointer to the created entry input stream, or nullptr on failure
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
  /// Removes any existing entry with the given path (TAR last-wins semantics).
  /// Uses standard erase-remove idiom.
  /// @param path - Entry path to remove duplicates for
  void remove_duplicate_entry(const std::string& path);

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
} // namespace morphizen
