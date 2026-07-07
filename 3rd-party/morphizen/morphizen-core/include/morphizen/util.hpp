/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include "morphizen/temp_file_stream.hpp"
#include <filesystem>
#include <iostream>
#include <memory>
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#include <sstream>
#include <vector>
#ifdef _WIN32
#  define fseek64 _fseeki64
#  define ftell64 _ftelli64
#else
#  define fseek64 fseeko
#  define ftell64 ftello
#endif

namespace morphizen {
MORPHIZEN_DLL_SPEC void dump_graph(const Graph& graph,
                                   const std::string& filename);
template <typename T> std::string container_as_string(const T& container) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& v : container) {
    if (c != 0) {
      str << ",";
    }
    str << v;
    c = c + 1;
  }
  str << "]";
  return str.str();
}

std::string find_file_in_path(const std::string& file, const char* env_name,
                              bool required);
std::string slurp(const char* filename);
MORPHIZEN_DLL_SPEC std::string slurp(const std::filesystem::path& path);
std::string slurp_if_exists(const std::filesystem::path& path);

MORPHIZEN_DLL_SPEC std::unique_ptr<int> scale_to_fix_point(float scale);
#ifdef ENABLE_PYTHON
MORPHIZEN_DLL_SPEC std::shared_ptr<void> init_interpreter();
MORPHIZEN_DLL_SPEC void eval_python_code(const std::string& code);
#endif
MORPHIZEN_DLL_SPEC std::filesystem::path get_morphizen_path();

#ifdef _WIN32
#  include <cstdio>
/**
 * Creates a temporary file with POSIX delete semantics for crash-resilient
 * cleanup.
 *
 * This function creates a temporary file using tmpfile_s() and then applies
 * POSIX delete semantics, which moves the file to $Extend\$Deleted immediately.
 * This ensures the file is automatically cleaned up on the next volume mount,
 * even if the system crashes.
 *
 * The function gracefully falls back to standard tmpfile_s() behavior if:
 * - POSIX delete is disabled via MORPHIZEN_ENABLE_POSIX_DELETE=0
 * - The Windows version doesn't support POSIX delete (requires Windows 10
 * 1809+)
 * - Any error occurs during POSIX delete setup
 *
 * @return FILE* pointer to the temporary file, or nullptr on failure.
 *         The FILE* remains fully functional regardless of whether POSIX delete
 * was applied.
 */
MORPHIZEN_DLL_SPEC FILE* tmpfile_with_posix_delete();
#endif // _WIN32

/// Creates a temporary file using platform-specific tmpfile implementation.
/// On Windows, uses tmpfile_with_posix_delete() for better cleanup behavior.
/// On other platforms, uses standard std::tmpfile().
/// @return FILE* pointer to temporary file, or nullptr on failure.
///         Callers MUST check for nullptr and handle errors appropriately.
inline FILE* create_tmpfile() {
#ifdef _WIN32
  return tmpfile_with_posix_delete();
#else
  return std::tmpfile();
#endif
}

/**
 * Converts a string from DOS/Windows format to Unix format.
 *
 * @param input The input string to be converted.
 * @return The converted string in Unix format.
 *
 * @note when we use PassContext::read_file, the format is always binary format,
 * on Windows, potentially a newline is encoded with `\r\n`, this function
 * convert it back to text format.
 */
MORPHIZEN_DLL_SPEC std::string dos2unix(const gsl::span<const char> input);

/**
 * Reads the contents of a binary file into a vector of uint8_t.
 *
 * @param filename The path to the binary file.
 * @return A vector of uint8_t containing the contents of the file.
 */
MORPHIZEN_DLL_SPEC std::vector<uint8_t>
slurp_binary_u8(const std::filesystem::path& filename);
MORPHIZEN_DLL_SPEC std::vector<int8_t>
slurp_binary_i8(const std::filesystem::path& filename);
MORPHIZEN_DLL_SPEC std::vector<char>
slurp_binary_c8(const std::filesystem::path& filename);
/**
 * Writes the binary data to the specified file.
 *
 * @param filename The path to the file where the binary data will be dumped.
 * @param data The binary data to be dumped.
 *
 * @return true if the data was successfully dumped, false otherwise.
 */
MORPHIZEN_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                                    gsl::span<const uint8_t> data);
MORPHIZEN_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                                    gsl::span<const int8_t> data);
MORPHIZEN_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                                    gsl::span<const char> data);
unsigned int get_tid();
unsigned int get_pid();

// Stream utility functions (for copying and filtering streams)
inline void stream_copy(std::istream& src, std::ostream& dst,
                        size_t buffer_size = 8192) {
  std::vector<char> buffer(buffer_size);
  while (src.read(buffer.data(), buffer_size) || src.gcount() > 0) {
    dst.write(buffer.data(), src.gcount());
  }
}

// Helper class that owns TempFileStream and provides istream interface
class TempFileStreamIstream : public std::istream {
public:
  explicit TempFileStreamIstream(std::unique_ptr<TempFileStream> temp)
      : std::istream(temp->get_read_stream().rdbuf()), temp_(std::move(temp)) {}

private:
  std::unique_ptr<TempFileStream> temp_;
};

template <typename F, typename... Args>
inline std::unique_ptr<std::istream>
stream_filter(std::istream& src, const F& filter, Args&&... args) {
  auto temp = std::make_unique<TempFileStream>();
  filter(src, temp->get_write_stream(), std::forward<Args>(args)...);
  temp->get_write_stream().flush();
  return std::make_unique<TempFileStreamIstream>(std::move(temp));
}

MORPHIZEN_DLL_SPEC std::unique_ptr<std::istream>
context_cache_files_to_tar_stream(class PassContext& context);

// TODO: defined morphizen_compile_model.cpp
MORPHIZEN_DLL_SPEC std::string
get_md5_of_file(const std::filesystem::path& path);
MORPHIZEN_DLL_SPEC std::string get_md5_of_buffer(const char* buffer,
                                                 size_t size);
} // namespace morphizen
