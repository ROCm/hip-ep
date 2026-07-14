/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace morphizen {

/**
 * @brief Abstract interface for stream-based file reading
 *
 * FileReader provides a generic abstraction for reading file data in a
 * streaming fashion, enabling efficient processing of large files without
 * loading entire contents into memory.
 *
 * This interface is particularly useful for:
 * - Processing large files with limited memory
 * - Inter-DLL data transfer without serialization overhead
 * - Pipeline-style data processing (e.g., compile(FileReader*, FileWriter*))
 * - Abstracting file sources (disk, memory, network, etc.)
 *
 * @par Design Rationale
 * Unlike std::istream, this interface is minimal and focused on binary data
 * transfer, making it ideal for DLL boundaries and simple streaming scenarios.
 * The optional mmap() support enables zero-copy access when available.
 *
 * @par Thread Safety
 * Implementations must document their thread-safety guarantees.
 * The interface itself does not mandate thread-safe behavior.
 *
 * @par Example Usage
 * @code
 * void process_file(const FileReader& reader) {
 *     char buffer[4096];
 *     reader.rewind();
 *
 *     while (true) {
 *         size_t bytes_read = reader.fread(buffer, sizeof(buffer));
 *         if (bytes_read == 0) break;
 *         // Process buffer...
 *     }
 * }
 * @endcode
 *
 * @par Zero-Copy Access
 * @code
 * void process_mmap(const FileReader& reader) {
 *     void* data = reader.mmap();
 *     if (data != nullptr) {
 *         // Direct memory access, no copy
 *         process_data(data, reader.size());
 *     } else {
 *         // Fallback to streaming read
 *         process_file(reader);
 *     }
 * }
 * @endcode
 */
class FileReader {
public:
  FileReader() = default;
  FileReader(const FileReader &) = delete;
  FileReader &operator=(const FileReader &) = delete;
  virtual ~FileReader() = default;

  /**
   * @brief Returns the total size of the file in bytes
   *
   * @return Size of the file in bytes
   *
   * @note The size should remain constant for the lifetime of the FileReader
   */
  virtual size_t size() const = 0;

  /**
   * @brief Resets the read position to the beginning of the file
   *
   * After calling rewind(), the next fread() call will read from the start
   * of the file.
   *
   * @note This operation should be relatively cheap (O(1) if possible)
   */
  virtual void rewind() const = 0;

  /**
   * @brief Reads data from the current position into the provided buffer
   *
   * Reads up to 'size' bytes from the current file position into 'buffer'.
   * The read position advances by the number of bytes actually read.
   *
   * @param buffer Pointer to destination buffer (must be at least 'size' bytes)
   * @param size Maximum number of bytes to read
   *
   * @return Number of bytes actually read (may be less than 'size' at EOF)
   *         Returns 0 when at end of file
   *
   * @pre buffer must point to valid memory of at least 'size' bytes
   * @post Read position advances by the return value
   *
   * @par Behavior at EOF
   * When the read position reaches the end of the file, subsequent calls
   * return 0 until rewind() is called.
   */
  virtual std::size_t fread(void *buffer, std::size_t size) const = 0;

  /**
   * @brief Provides direct memory-mapped access to file contents (optional)
   *
   * Returns a pointer to the file's data in memory, enabling zero-copy access.
   * If memory-mapping is not available or not supported, returns nullptr.
   *
   * @return Pointer to memory-mapped file data, or nullptr if unavailable
   *
   * @note The returned pointer remains valid for the lifetime of the FileReader
   * @note Caller must not modify the data (treat as const)
   * @note Use size() to determine the size of the mapped region
   *
   * @par When to use
   * - Large files where multiple passes are needed
   * - Random access patterns
   * - Performance-critical code paths
   *
   * @par Fallback pattern
   * Always provide a fallback to fread() when mmap() returns nullptr,
   * as not all implementations support memory mapping.
   */
  virtual void *mmap() { return nullptr; }
};

/**
 * @brief Abstract interface for stream-based file writing
 *
 * FileWriter provides a generic abstraction for writing file data in a
 * streaming fashion, enabling efficient generation of large files without
 * buffering entire contents in memory.
 *
 * This interface is particularly useful for:
 * - Generating large output files with limited memory
 * - Inter-DLL data transfer without serialization overhead
 * - Pipeline-style data processing (e.g., compile(FileReader*, FileWriter*))
 * - Abstracting file destinations (disk, memory, network, etc.)
 *
 * @par Design Rationale
 * Unlike std::ostream, this interface is minimal and focused on binary data
 * transfer, making it ideal for DLL boundaries and simple streaming scenarios.
 *
 * @par Thread Safety
 * Implementations must document their thread-safety guarantees.
 * The interface itself does not mandate thread-safe behavior.
 *
 * @par Example Usage
 * @code
 * void generate_file(FileWriter& writer) {
 *     char buffer[4096];
 *     // Fill buffer with data...
 *     writer.fwrite(buffer, sizeof(buffer));
 * }
 * @endcode
 *
 * @par Pipeline Pattern
 * @code
 * void compile(const FileReader& input, FileWriter& output) {
 *     char buffer[4096];
 *     input.rewind();
 *
 *     while (true) {
 *         size_t bytes = input.fread(buffer, sizeof(buffer));
 *         if (bytes == 0) break;
 *
 *         // Transform data...
 *
 *         output.fwrite(buffer, bytes);
 *     }
 * }
 * @endcode
 */
class FileWriter {
public:
  FileWriter() = default;
  FileWriter(const FileWriter &) = delete;
  FileWriter &operator=(const FileWriter &) = delete;
  virtual ~FileWriter() = default;

  /**
   * @brief Writes data from the provided buffer to the file
   *
   * Writes 'size' bytes from 'buffer' to the current position in the file.
   * The write position advances by the number of bytes written.
   *
   * @param buffer Pointer to source data (must be at least 'size' bytes)
   * @param size Number of bytes to write
   *
   * @return Number of bytes actually written (should equal 'size' on success)
   *
   * @pre buffer must point to valid memory of at least 'size' bytes
   * @post Write position advances by the return value
   *
   * @throws Implementation-defined exceptions on write failure
   *
   * @par Error Handling
   * Implementations should clearly document their error handling strategy:
   * - Return value < size indicates partial write
   * - May throw exceptions for unrecoverable errors
   * - Should provide diagnostic information on failure
   *
   * @par Buffering
   * Implementations may buffer writes internally for performance.
   * Ensure data is flushed before the FileWriter is destroyed.
   */
  virtual std::size_t fwrite(const void *buffer, std::size_t size) const = 0;
};

/**
 * @brief Abstract interface for file system operations
 *
 * FileSystem provides a DLL-safe factory for creating FileReader and FileWriter
 * instances. It decouples file I/O creation from any specific context (such as
 * PassContext), allowing generic components to open files without depending on
 * higher-level abstractions.
 *
 * @par DLL Safety
 * - Uses const char* for paths (no std::string across DLL boundaries)
 * - Returns raw pointers; caller owns the returned object and must delete it
 * - Both FileReader and FileWriter have virtual destructors, so delete through
 *   the base pointer correctly invokes the derived destructor
 *
 * @par Example Usage (raw pointers with manual destroy)
 * @code
 * void process(FileSystem& fs) {
 *     FileReader* reader = fs.create_reader("input.bin");
 *     FileWriter* writer = fs.create_writer("output.bin");
 *     if (reader && writer) {
 *         char buf[4096];
 *         reader->rewind();
 *         while (auto n = reader->fread(buf, sizeof(buf)))
 *             writer->fwrite(buf, n);
 *     }
 *     fs.destroy_writer(writer);
 *     fs.destroy_reader(reader);
 * }
 * @endcode
 *
 * @par Example Usage (RAII with create_reader_template/create_writer_template)
 * @code
 * void process(FileSystem& fs) {
 *     auto reader = fs.create_reader_template("input.bin");
 *     auto writer = fs.create_writer_template("output.bin");
 *     if (reader && writer) {
 *         char buf[4096];
 *         reader->rewind();
 *         while (auto n = reader->fread(buf, sizeof(buf)))
 *             writer->fwrite(buf, n);
 *     }
 *     // automatically destroyed when reader/writer go out of scope
 * }
 * @endcode
 */
class FileSystem {
public:
  FileSystem() = default;
  FileSystem(const FileSystem &) = delete;
  FileSystem &operator=(const FileSystem &) = delete;
  virtual ~FileSystem() = default;

  /**
   * @brief Creates a FileReader for the given path
   *
   * @param path Null-terminated file path
   * @return Pointer to a new FileReader, or nullptr on failure.
   *         Caller must pass the returned pointer to destroy_reader() when
   *         done.
   */
  virtual FileReader *create_reader(const char *path) = 0;

  /**
   * @brief Creates a FileWriter for the given path
   *
   * @param path Null-terminated file path
   * @return Pointer to a new FileWriter, or nullptr on failure.
   *         Caller must pass the returned pointer to destroy_writer() when
   *         done.
   */
  virtual FileWriter *create_writer(const char *path) = 0;

  /**
   * @brief Destroys a FileReader previously returned by create_reader()
   *
   * Ensures deallocation happens in the same DLL that allocated the object,
   * avoiding cross-DLL heap mismatches.
   *
   * @param reader Pointer to destroy (nullptr is safely ignored)
   */
  virtual void destroy_reader(FileReader *reader) = 0;

  /**
   * @brief Destroys a FileWriter previously returned by create_writer()
   *
   * Ensures deallocation happens in the same DLL that allocated the object,
   * avoiding cross-DLL heap mismatches.
   *
   * @param writer Pointer to destroy (nullptr is safely ignored)
   */
  virtual void destroy_writer(FileWriter *writer) = 0;

  template <typename T> struct Deleter {
    FileSystem *fs;
    void operator()(T *ptr) const {
      if constexpr (std::is_same_v<T, FileReader>)
        fs->destroy_reader(ptr);
      else
        fs->destroy_writer(ptr);
    }
  };

  std::unique_ptr<FileReader, Deleter<FileReader>>
  create_reader_template(const char *path) {
    return {create_reader(path), {this}};
  }

  std::unique_ptr<FileWriter, Deleter<FileWriter>>
  create_writer_template(const char *path) {
    return {create_writer(path), {this}};
  }
};

} // namespace morphizen
