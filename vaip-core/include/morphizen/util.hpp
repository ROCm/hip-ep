/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "./_sanity_check.hpp"
#include <filesystem>
#include <morphizen/vaip_io.hpp>
#include <sstream>
#include <vaip/my_ort.h>
#include <vaip/vaip_gsl.h>
#ifdef _WIN32
#  define fseek64 _fseeki64
#  define ftell64 _ftelli64
#else
#  define fseek64 fseeko
#  define ftell64 ftello
#endif

namespace vaip_core {
VAIP_DLL_SPEC void dump_graph(const Graph& graph, const std::string& filename);
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

VAIP_DLL_SPEC std::string convert_to_xir_op_type(const std::string& domain,
                                                 const std::string& op_type);

std::string find_file_in_path(const std::string& file, const char* env_name,
                              bool required);
std::string slurp(const char* filename);
VAIP_DLL_SPEC std::string slurp(const std::filesystem::path& path);
std::string slurp_if_exists(const std::filesystem::path& path);

VAIP_DLL_SPEC std::unique_ptr<int> scale_to_fix_point(float scale);
#ifdef ENABLE_PYTHON
VAIP_DLL_SPEC std::shared_ptr<void> init_interpreter();
VAIP_DLL_SPEC void eval_python_code(const std::string& code);
#endif
VAIP_DLL_SPEC std::filesystem::path get_vaip_path();

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
VAIP_DLL_SPEC std::string dos2unix(const gsl::span<const char> input);

/**
 * Reads the contents of a binary file into a vector of uint8_t.
 *
 * @param filename The path to the binary file.
 * @return A vector of uint8_t containing the contents of the file.
 */
VAIP_DLL_SPEC std::vector<uint8_t>
slurp_binary_u8(const std::filesystem::path& filename);
VAIP_DLL_SPEC std::vector<int8_t>
slurp_binary_i8(const std::filesystem::path& filename);
VAIP_DLL_SPEC std::vector<char>
slurp_binary_c8(const std::filesystem::path& filename);
/**
 * Writes the binary data to the specified file.
 *
 * @param filename The path to the file where the binary data will be dumped.
 * @param data The binary data to be dumped.
 *
 * @return true if the data was successfully dumped, false otherwise.
 */
VAIP_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                               gsl::span<const uint8_t> data);
VAIP_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                               gsl::span<const int8_t> data);
VAIP_DLL_SPEC bool dump_binary(const std::filesystem::path& filename,
                               gsl::span<const char> data);
/**
 * Compresses the given data using a specified compression level.
 *
 * @param data The data to be compressed.
 * @param level The compression level (default is 9).
 * @return The compressed data as a vector of uint8_t.
 */
VAIP_DLL_SPEC std::vector<uint8_t> compress(gsl::span<const uint8_t> data,
                                            int level = 9);
VAIP_DLL_SPEC std::vector<int8_t> compress(gsl::span<const int8_t> data,
                                           int level = 9);
VAIP_DLL_SPEC std::vector<char> compress(gsl::span<const char> data,
                                         int level = 9);
/**
 * @brief Uncompresses the given data.
 *
 * This function takes a span of uint8_t data and uncompresses it, returning the
 * uncompressed data as a std::vector<uint8_t>.
 *
 * @param data The data to be uncompressed.
 * @return The uncompressed data as a std::vector<uint8_t>.
 */
VAIP_DLL_SPEC std::vector<uint8_t> uncompress(gsl::span<const uint8_t> data);
VAIP_DLL_SPEC std::vector<int8_t> uncompress(gsl::span<const int8_t> data);
VAIP_DLL_SPEC std::vector<char> uncompress(gsl::span<const char> data);

unsigned int get_tid();
unsigned int get_pid();

VAIP_DLL_SPEC void compress(const IStreamReader& src, IStreamWriter& dst,
                            int compress_level = 9);
VAIP_DLL_SPEC std::unique_ptr<IStreamReader> compress(const IStreamReader& src,
                                                      int compress_level = 9);
VAIP_DLL_SPEC void uncompress(const IStreamReader& src, IStreamWriter& dst);
VAIP_DLL_SPEC std::unique_ptr<IStreamReader>
uncompress(const IStreamReader& src);

VAIP_DLL_SPEC std::unique_ptr<IStreamReader>
context_cache_files_to_tar_stream(class PassContext& context);

VAIP_DLL_SPEC std::unique_ptr<IStreamReader>
context_tar_file_to_tar_stream(class PassContext& context);

// TODO: defined vitisai_compile_model.cpp
VAIP_DLL_SPEC std::string get_md5_of_file(const std::filesystem::path& path);
VAIP_DLL_SPEC std::string get_md5_of_buffer(const char* buffer, size_t size);
} // namespace vaip_core
