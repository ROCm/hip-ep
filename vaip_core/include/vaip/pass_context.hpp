/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#pragma once

#include "./vaip_io.hpp"
#include "vaip/pass_context.pb.h"
#include "vitis/ai/env_config.hpp"
#include <filesystem>
#include <gsl/span>
#include <memory>
#include <optional>
namespace vaip_core {
// The reason PassContext exists is that PassContext has a longer life cycle
// than Pass. The Pass will be destoryed after model is compiled but some info
// is still needed for custom op.
// clang-format off
/**



                              write_file     read_file

                                   │           ▲
                                   │           │
       cache_files_to_tar_mem      │           │     tar_file_to_cache_files
                                   ▼           │
   ┌─────────┐    ◄──────     ┌────────────────┴─────┐   ◄────── ┌─────────┐
   │ tar mem │                │   memory cache files │           │ tar file│
   └─────────┘    ──────►     └──────────────────────┘   ──────► └─────────┘

       tar_mem_to_cache_files     │            ▲   cache_files_to_tar_file
                                  │            │
                                  │            │
                                  │            │
                                  ▼            │
                    cache_files_to_dir        dir_to_cache_files


*/
// clang-format on
class PassContextTimer {
public:
  PassContextTimer();
  virtual ~PassContextTimer();
};
class CacheFileReader {
public:
  CacheFileReader() = default;
  CacheFileReader(const CacheFileReader&) = delete;
  virtual ~CacheFileReader() = default;
  virtual size_t size() const = 0;
  virtual std::size_t fread(void* buffer, std::size_t size) const = 0;
};
class CacheFileWriter {
public:
  CacheFileWriter() = default;
  CacheFileWriter(const CacheFileWriter&) = delete;
  virtual ~CacheFileWriter() = default;
  virtual std::size_t fwrite(const void* buffer, std::size_t size) const = 0;
};
class PassContext {
public:
public:
  virtual ~PassContext() = default;
  /**
   * @brief Gets the directory path where log files are stored.
   *
   * @return The directory path where log files are stored.
   *
   * This directory is the cache directory.
   */
  virtual std::filesystem::path get_log_dir() const = 0;
  /**
   * Retrieves the value of a provider option based on the given option name.
   *
   * @param option_name The name of the option to retrieve.
   * @return An optional string containing the value of the option if found, or
   * an empty optional if the option does not exist.
   */
  virtual std::optional<std::string>
  get_provider_option(const std::string& option_name) const = 0;
  /**
   * Retrieves the value of a session config based on the given option name.
   *
   * @param option_name The name of the option to retrieve.
   * @return An optional string containing the value of the option if found, or
   * an empty optional if the option does not exist.
   */
  virtual std::optional<std::string>
  get_session_config(const std::string& option_name) const = 0;
  /**
   * Retrieves the value of a provider option.
   *
   * This function retrieves the value of a provider option specified by the
   * given option name. If the option is not found, it returns the default value
   * provided.
   *
   * @param option_name The name of the option to retrieve.
   * @param default_value The default value to return if the option is not
   * found.
   * @return The value of the option if found, otherwise the default value.
   */
  virtual std::string
  get_provider_option(const std::string& option_name,
                      const std::string& default_value) const = 0;
  virtual bool cache_in_mem() const = 0;
/**
 * @brief Helper macro to get provider option with class.
 *
 * This macro simplifies the process of retrieving a provider option by using
 * the class name of the environment parameter.
 *
 * @code
 * DEF_ENV_PARAM_2(YOUR_PROVIDER_OPTION_NAME, "<default-value>", int64_t)
 * int64_t value =
 *    VAIP_PROVIDER_OPTION(*pass.get_context(), YOUR_PROVIDER_OPTION_NAME)
 *
 * DEF_ENV_PARAM_2(YOUR_BOOLEN_OPTION, "ON", bool)
 * bool value =
 *    VAIP_PROVIDER_OPTION(*pass.get_context(), YOUR_BOOLEN_OPTION)
 *
 * DEF_ENV_PARAM(YOUR_INT_OPTION, "100") // int value
 * int value =
 *    VAIP_PROVIDER_OPTION(*pass.get_context(), YOUT_INT_OPTION)
 * @endcode
 *
 * now we can use environment variable as the default value of a
 * provider option.
 *
 * Users can overwrite the default provider options by exciplictly set the
 * environment variable or overwrite the provider option in C++ or Python.
 *
 *
 * @param context The context object to retrieve the option from.
 * @param param_name The name of the parameter to retrieve.
 * @return The value of the provider option.
 */
#define VAIP_PROVIDER_OPTION(context, param_name)                              \
  ((context).get_provier_option_with_class<ENV_PARAM_##param_name>())

  /**
   * @brief Retrieves the value of a provider option using a class.
   *
   * This template function retrieves the value of a provider option using the
   * class name of the environment parameter. It converts the retrieved string
   * value to the appropriate type.
   *
   * @tparam env_name The class name of the environment parameter.
   * @return The value of the provider option converted to the appropriate type.
   */
  template <typename env_name>
  decltype(env_name::value) get_provier_option_with_class() const {
    const char* name = env_name::get_name();
    const char* defvalue = env_name::get_default_value();
    auto p = get_provider_option(std::string(name), std::string(defvalue));
    return vitis::ai::env_config_helper<decltype(env_name::value)>::from_string(
        p);
  }

  /**
   * Retrieves the value of a session configuration.
   *
   * This function retrieves the value of a session config specified by the
   * given option name. If the option is not found, it returns the default value
   * provided.
   *
   * @param option_name The name of the configuration to retrieve.
   * @param default_value The default value to return if the config is not
   * found.
   * @return The value of the option if found, otherwise the default value.
   */
  virtual std::string
  get_session_config(const std::string& option_name,
                     const std::string& default_value) const = 0;
  virtual std::string
  get_run_option(const std::string& option_name,
                 const std::string& default_value) const = 0;
  /**
   * @brief Retrieves the configuration protobuf object.
   *
   * This function returns a reference to the configuration protobuf object
   * associated with the pass context.
   *
   * @return A constant reference to the configuration protobuf object.
   *
   * @sa config.proto
   *
   * @note this is the low level configuration, it is not recommended to use it,
   * please use `get_log_dir` or `get_provider_options` if possible.
   */
  virtual const ConfigProto& get_config_proto() const = 0;
  // @brief DO NOT USE THIS FUNCTION
  virtual std::shared_ptr<void>
  get_context_resource(const std::string& name) const = 0;

  /**
   * @brief Reads in-memory cache files into bytes
   *
   * @param filename The name of the file to be read.
   * @return The contents of the file as a std::optional<std::vector<char>>.
   *         Returns std::nullopt if the filename is not found.
   *
   */
  virtual std::optional<std::vector<char>>
  read_file_c8(const std::string& filename) const = 0;

  virtual std::optional<std::vector<uint8_t>>
  read_file_u8(const std::string& filename) const = 0;

  virtual std::unique_ptr<CacheFileReader>
  open_file_for_read(const std::string& filename) const = 0;
  virtual std::unique_ptr<CacheFileWriter>
  open_file_for_write(const std::string& filename) = 0;

  virtual FILE* open_file(const std::string& filename) const = 0;

  /**
   * @brief Saves the filename and its data into in-memory cache files
   *
   * @param filename The name of the file to write to.
   * @param data A gsl::span<const char> representing the data to be written.
   * @return True if the file was successfully written, false otherwise.
   *
   */
  virtual bool write_file(const std::string& filename,
                          gsl::span<const char> data) = 0;

  /**
   * restore cache_files_ from saved file
   */
  virtual void restore_cache_files() = 0;

  /**
   * @brief Checks if a cache file with the given filename exists.
   *
   * @param filename The name of the cache file to check.
   * @return True if the cache file exists, false otherwise.
   */
  virtual bool has_cache_file(const std::string& filename) const = 0;

  /**
   * Retrieves the names of cache files associated with the given filename.
   *
   * @param filename The name of the file.
   * @return A vector of strings containing the names of cache files.
   */
  virtual std::vector<std::string> get_cache_file_names() const = 0;
  /**
   * @brief Creates a tar file from in-memory cache files
   *
   * @param tar_file The path to the tar file.
   * @return True if the tar file was successfully created, false otherwise.
   *
   */
  virtual bool cache_files_to_tar_file(IStreamWriter& writer) const = 0;

  /**
   * @brief Loads a tar file and save its content into in-memory cache files
   *
   * @param tar_file The path to the tar file.
   * @return True if the tar ball was successfully loaded, false otherwise.
   *
   */
  virtual bool
  tar_file_to_cache_files(const std::filesystem::path& tar_file) = 0;

  /**
   * @brief Creates a in-memory tar file from in-memory cache files
   *
   * @return A std::vector<char representing the tar file.
   *
   */
  virtual std::vector<char> cache_files_to_tar_mem() const = 0;

  /**
   * @brief Loads a in-memory tar file save its content into in-memory cache
   * files
   *
   * @param data A pointer to the data containing the tar ball.
   * @param size The size of the data in bytes.
   * @return True if the in-memory cache files successfully created, false
   * otherwise.
   *
   */
  virtual bool tar_mem_to_cache_files(const char* data, size_t size) = 0;
  virtual bool tar_file_to_cache_files(FILE*) = 0;
  virtual bool tar_file_to_cache_files(class IStreamReader& src) = 0;

  /**
   * @brief Creates a new instance of PassContext.
   *
   * @return A unique pointer to the newly created PassContext object.
   */
  VAIP_DLL_SPEC static std::unique_ptr<PassContext> create();

  /**
   * @brief Dump xclbin to cache directory if not exists. Return the dumped
   * xclbin path
   *
   * @return The xclbin path.
   */
  virtual std::filesystem::path
  xclbin_path_to_cache_files(const std::filesystem::path& path) const = 0;
  /**
   * @brief Reads an xclbin file from the specified path.
   *
   * @param path The path to the xclbin file.
   * @return A span of const char representing the contents of the xclbin file.
   */
  virtual std::optional<std::vector<char>>
  read_xclbin(const std::filesystem::path& path) const = 0;
  /**
   * @brief collect time for profiling.
   */
  virtual std::unique_ptr<PassContextTimer>
  measure(const std::string& label) = 0;
  /**
   * Saves the context to `get_log_dir()/context.json`
   */
  virtual void save_context_json() const = 0;

  virtual void set_is_ep_context_model(bool is_ep_context_model) = 0;

  virtual bool get_is_ep_context_model() = 0;
};
} // namespace vaip_core
