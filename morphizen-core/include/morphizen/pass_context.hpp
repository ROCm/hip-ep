/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(                                                             \
          disable : 4946 4267) // reinterpret_cast / size_t→int in protobuf
#endif
#include "morphizen/pass_context.pb.h"
#ifdef _WIN32
#  pragma warning(pop)
#endif

#include <filesystem>
#include <gsl/span>
#include <iosfwd>
#include <memory>
#include <optional>

// Import generic file I/O interfaces from foundation
#include <morphizen-foundation/file_io.hpp>

namespace morphizen {
// The reason PassContext exists is that PassContext has a longer life cycle
// than Pass. The Pass will be destoryed after model is compiled but some info
// is still needed for custom op.
//
// PassContext provides access to cached compilation artifacts via tar_file_
// system. Use write_file() and read_file_*() for individual files.
class PassContextTimer {
public:
  PassContextTimer();
  virtual ~PassContextTimer();
};

// Type aliases for backwards compatibility with existing code
// These interfaces have been moved to morphizen-foundation as
// FileReader/FileWriter for generic reuse across the codebase and in DLL
// boundaries.
using CacheFileReader = FileReader;
using CacheFileWriter = FileWriter;

class QoSUpdateInterface {
public:
  virtual ~QoSUpdateInterface() = default;
  virtual void update_qos(const std::string& perf_pref_value) = 0;
};

class PassContext {
public:
public:
  virtual ~PassContext() = default;
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
 *    MORPHIZEN_PROVIDER_OPTION(*pass.get_context(), YOUR_PROVIDER_OPTION_NAME)
 *
 * DEF_ENV_PARAM_2(YOUR_BOOLEN_OPTION, "ON", bool)
 * bool value =
 *    MORPHIZEN_PROVIDER_OPTION(*pass.get_context(), YOUR_BOOLEN_OPTION)
 *
 * DEF_ENV_PARAM(YOUR_INT_OPTION, "100") // int value
 * int value =
 *    MORPHIZEN_PROVIDER_OPTION(*pass.get_context(), YOUT_INT_OPTION)
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
#define MORPHIZEN_PROVIDER_OPTION(context, param_name)                         \
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
   * @note this function is to be deprecated, please use get_provier_option,
   * only support XLNX_model_clone_external_data_threshold for backward
   * compatibility.
   */
  template <typename env_name>
  decltype(env_name::value) get_provier_option_with_class() const;
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
   * @brief Retrieves the meta definition parameter from the given MetaDefProto
   * object.
   *
   * @param meta_def A reference to a MetaDefProto object from which the meta
   * definition parameter is extracted.
   * @return A string representing the meta definition parameter.
   */
  virtual std::string
  get_meta_def_param(const MetaDefProto& meta_def) const = 0;
  virtual std::string
  get_ep_dynamic_option(const std::string& option_name,
                        const std::string& default_value) const = 0;

  virtual void remove_QosUpdater(QoSUpdateInterface* updater) = 0;
  virtual void
  add_QosUpdater(const std::shared_ptr<QoSUpdateInterface>& updater) const = 0;
  virtual void update_all_qos(const std::string& workload_type) const = 0;
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

  /**
   * @brief Pure virtual function to retrieve the context protobuf object.
   *
   * This function must be overridden by derived classes to provide
   * access to the context protocol object.
   *
   * @return A constant reference to the ContextProto object.
   */
  virtual const ContextProto& get_context_proto() const = 0;
  virtual ContextProto& get_context_proto() = 0;
  // @brief DO NOT USE THIS FUNCTION
  virtual std::shared_ptr<void>
  get_context_resource(const std::string& name) const = 0;

  virtual std::filesystem::path get_model_path() const = 0;
  /**
   * @brief Returns the directory path for debugging and troubleshooting dumps.
   *
   * This directory is used ONLY for debugging output files such as graph dumps
   * (.txt, .onnx), fix info, and diagnostic reports. It is NOT used for cache
   * persistence (which uses EP context tar-based cache).
   *
   * The dump directory can be overridden via the 'dump_dir' provider option.
   * Otherwise, defaults to: temp/morphizen_dumps/<cache_key>
   *
   * @return The directory path where dump files should be written.
   */
  virtual std::filesystem::path get_dump_directory() const = 0;
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
   * @brief Creates a new instance of PassContext.
   *
   * @return A unique pointer to the newly created PassContext object.
   */
  MORPHIZEN_DLL_SPEC static std::unique_ptr<PassContext> create();
  /**
   * @brief collect time for profiling.
   */
  virtual std::unique_ptr<PassContextTimer>
  measure(const std::string& label) = 0;
  /**
   * Saves the context to `get_log_dir()/context.json`
   */
  virtual void save_context_json() const = 0;

  virtual void on_custom_op_create_end() = 0;
  virtual std::map<std::string, std::string>
  get_all_provider_options() const = 0;

  /**
   * @brief Appends compatibility information for a compiled model.
   *
   *  This method is intended to be called by each level-1 pass during the model
   * compilation process. It records backend-specific compatibility information
   * that describes special requirements, limitations, or metadata associated
   * with the compiled model for a given backend.
   *
   * @param backend_name The name of the backend (e.g.,
   * "morphizen-pass_level1_dpu", "morphizen-pass_level1_dd_cxx",
   * "morphizen-pass_vaiml_partition") to which the compatibility information
   * applies.
   *
   * @param compatibility_info A string containing the compatibility details or
   * metadata for the specified backend.
   *
   */
  virtual void append_compiled_model_compatibility_info(
      const std::string& backend_name,
      const std::string& compatibility_info) = 0;
  /**
   * @brief Retrieves the compiled model compatibility information.
   * This method returns a const reference to the map of all backend names
   * to their associated compatibility information that has been recorded
   * during compilation.
   *
   * @return A const reference to a map where each key is a backend name
   *         and the corresponding value is its compatibility information
   * string. The reference remains valid as long as the PassContext exists.
   */
  virtual const std::map<std::string, std::string>&
  get_compiled_model_compatibility_info() const = 0;

  virtual void disable_delete_tar_file_in_session_created() = 0;
  virtual std::unique_ptr<FileSystem> get_file_system() = 0;
};
} // namespace morphizen
