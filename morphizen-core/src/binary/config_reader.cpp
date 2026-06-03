/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen_ort_api.h"
#include "morphizen/plugin.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_CONFIG_READER, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_CONFIG_READER) >= n)
DEF_ENV_PARAM_2(MORPHIZEN_CONFIG_PROVIDER_BACKEND, "onnxruntime_morphizen_ep",
                std::string)
// this is actually provider options, for backward compatibility, we keep
// this key in the root of the json.
static constexpr char kProviderOptions[] = "sessionOptions";
static constexpr char kSessionConfig[] = "ort_session_config";
static constexpr char kSessionOptionPtr[] = "session_options";
static constexpr char kEpProviderOptionPrefix[] =
    "ep.morphizenexecutionprovider.";
namespace morphizen {

namespace config_default {
#include "config_json_binary.hpp"
}

static const char* get_default_config() {
  // `with_default_morphizen_config` and `config` are generated
  // automatically by
  // ${CMAKE_CURRENT_SOURCE_DIR}/src/binary/config_json_binary.hpp.py
  if (config_default::with_default_morphizen_config) {
    return (const char*)&config_default::config[0];
  }
  return nullptr;
}

static void JsonFileToMessage(const std::string& file_path,
                              google::protobuf::Message* message) {
  std::ifstream input(file_path);
  if (!input.is_open()) {
    std::string error_message = "Failed to open file: " + file_path;
    MY_LOG(1) << error_message;
    throw std::runtime_error(error_message);
  }

  std::string json_content((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  google::protobuf::util::JsonParseOptions options;
  auto status = google::protobuf::util::JsonStringToMessage(json_content,
                                                            message, options);

  if (!status.ok()) {
    std::string error_message =
        "Failed to parse JSON: " + std::string(status.message().data());
    MY_LOG(1) << error_message;
    throw std::runtime_error(error_message);
  }

  return; // Return the successful status
}
static void set_struct_value(google::protobuf::Struct& struct_value,
                             const std::string& key1, const std::string& key2,
                             const std::string& string_value) {
  auto field1 = struct_value.mutable_fields()->find(key1);
  if (field1 == struct_value.mutable_fields()->end()) {
    field1 = struct_value.mutable_fields()->insert({key1, {}}).first;
  }
  auto fields2 = field1->second.mutable_struct_value()->mutable_fields();
  auto field2 = fields2->find(key2);
  if (field2 == fields2->end()) {
    field2 = fields2->insert({key2, {}}).first;
  }
  field2->second.set_string_value(string_value);
}
static std::unique_ptr<google::protobuf::Struct>
get_protobuf_struct_from_config_file(const std::string& filename) {
  std::ifstream f(filename);
  // parse the json file into Struct message
  auto config = std::make_unique<google::protobuf::Struct>();
  JsonFileToMessage(filename, config.get());
  return config;
}

// Removed: update_enable_batch - NPU-specific xlnx_enable_batch option

// Removed: update_num_dpu_runners - NPU-specific DPU runners configuration

static void set_session_config(google::protobuf::Struct& ret,
                               const std::string& key,
                               const std::string& value) {
  if (key.rfind(kEpProviderOptionPrefix, 0) == 0) {
    auto key2 = key.substr(sizeof(kEpProviderOptionPrefix) - 1);
    MY_LOG(1) << "convert " << key << " to " << key2 << " and set "
              << "provider_options"
              << "[" << key2 << "]=\"" << value << "\"";
    // The key is prefixed with "ep.morphizenexecutionprovider."
    // Remove the prefix before setting the value.
    set_struct_value(ret, kProviderOptions, key2, value);
  } else {
    MY_LOG(1) << "set " << kSessionConfig << "[" << key << "]=\"" << value
              << "\"";
    set_struct_value(ret, kSessionConfig, key, value);
  }
}

static void restore_session_options(google::protobuf::Struct& ret,
                                    std::string entry_second) {
  std::map<std::string, std::string> session_config_options_entry_list = {};
#if MORPHIZEN_ORT_API_MAJOR >= 10
  auto options = reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(entry_second));
  MORPHIZEN_ORT_API(session_option_configuration)
  (&session_config_options_entry_list, options,
   [](void* mmap, const char* name, const char* value) {
     auto* map_ptr =
         reinterpret_cast<std::map<std::string, std::string>*>(mmap);
     map_ptr->insert({name, value});
   });
#else
  LOG(WARNING) << "ORT API version is less than 10, now used is "
               << MORPHIZEN_ORT_API_MAJOR
               << ", session options will not be restored.";
#endif
  for (const auto& option : session_config_options_entry_list) {
    set_session_config(ret, option.first, option.second);
  }
}

static google::protobuf::Struct
get_config_json(const onnxruntime::ProviderOptions& options) {
  google::protobuf::Struct ret;
  // update_log_level(options);
  auto morphizen_get_default_config_plugin =
      ::morphizen::Plugin::get(ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND));
  const char* default_config = get_default_config();
  if (default_config == nullptr) {
    if (morphizen_get_default_config_plugin) {
      MY_LOG(1) << "found plugin: "
                << ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND);
      auto morphizen_get_default_config =
          morphizen_get_default_config_plugin->get_method<const char*>(
              "morphizen_get_default_config");
      if (morphizen_get_default_config) {
        MY_LOG(1) << "found symbol: morphizen_get_default_config from "
                  << ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND);
        default_config = morphizen_get_default_config();
      } else {
        MY_LOG(1) << "cannot found symbol: morphizen_get_default_config from "
                  << ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND);
      }
    } else {
      MY_LOG(1) << "cannot found plugin: "
                << ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND)
                << " fall back to builtin default";
    }
  }
  auto iterator_config_file = options.find("config_file");
  auto opt_config_file = std::optional<std::filesystem::path>();
  if (iterator_config_file != options.end()) {
    MY_LOG(1) << "found config_file in provider options: "
              << iterator_config_file->second;
    auto tmp_opt_config_file =
        std::filesystem::path(iterator_config_file->second);
    if (std::filesystem::exists(tmp_opt_config_file)) {
      opt_config_file = tmp_opt_config_file;
    } else {
      LOG(WARNING) << "config_file does not exist: "
                   << iterator_config_file->second
                   << " fall back to default config";
    }
  }
  if (opt_config_file.has_value()) {
    std::string config_file = options.at("config_file");
    MY_LOG(1) << " overwrite default config, read if from "
              << opt_config_file.value();
    auto struct_from_config_file =
        get_protobuf_struct_from_config_file(opt_config_file.value().string());
    if (struct_from_config_file == nullptr) {
      LOG(FATAL) << "failed to parse config file: " << opt_config_file.value();
    }
    ret = std::move(*struct_from_config_file);
  } else {
    MY_LOG(1) << "use default config";
    if (default_config == nullptr) {
      LOG(FATAL) << "no default morphizen_config.json, "
                    "provider_options[\"config_file\"] is required";
    }
    if (ENV_PARAM(MORPHIZEN_DEBUG_CONFIG_READER)) {
      auto stream = std::istringstream(default_config);
      while (stream.good()) {
        std::string line;
        std::getline(stream, line);
        MY_LOG(2) << line;
      }
    }
    auto status =
        google::protobuf::util::JsonStringToMessage(default_config, &ret);
    if (!status.ok()) {
      std::string err_msg =
          std::string{"failed to parse default config: "} + default_config;
      err_msg += "\n" + status.ToString();
      LOG(FATAL) << err_msg;
    }
  }
  const std::string ort_session_config_prefix =
      std::string(kSessionConfig) + ".";

  for (const auto& entry : options) {
    MY_LOG(1) << "process provider_option[\"" << entry.first << "\"]= \""
              << entry.second << "\"";
    if (entry.first == kSessionOptionPtr) {
      // The key here is "session_options," and the value is a string that
      // points to the session_options object.
      restore_session_options(ret, entry.second);
    } else if (entry.first.rfind(ort_session_config_prefix, 0) == 0) {
      set_session_config(ret,
                         entry.first.substr(ort_session_config_prefix.size()),
                         entry.second);
    } else {
      /* MY_LOG(1) << "set "
                << "provider_options"
                << "[" << entry.first << "]=\"" << entry.second << "\"";
      set_struct_value(ret, kProviderOptions, entry.first, entry.second);*/
    }
  }
  return ret;
}
extern "C" char** environ;
static std::unordered_map<std::string, std::string>
get_environment_variables() {
  std::unordered_map<std::string, std::string> env_map;
  for (auto env = environ; *env; ++env) {
    MY_LOG(3) << "get environment variable: " << *env;
    std::string key_value = *env;
    size_t pos = key_value.find('=');
    if (pos != std::string::npos) {
      std::string key = key_value.substr(0, pos);
      std::string value = key_value.substr(pos + 1);
      env_map[key] = value;
    }
  }
  return env_map;
}
static const onnxruntime::ProviderOptions get_provider_option_from_env_variable(
    const onnxruntime::ProviderOptions& options) {
  // enumerate all environment variables and check if the variable name start
  // with "MORPHIZEN_EP_PROVIER_OPTION."
  const std::string prefix = "MORPHIZEN_EP_PROVIDER_OPTION_";
  onnxruntime::ProviderOptions ret = options;
  for (const auto& entry : get_environment_variables()) {
    if (entry.first.rfind(prefix, 0) == 0) {
      // Extract the option name and value from the environment variable
      std::string option_name = entry.first.substr(prefix.size());
      std::string option_value = entry.second;
      MY_LOG(1) << "set "
                << "provider_options"
                << "[\"" << option_name << "\"]=\"" << option_value
                << "\" from \"${ENV:" << entry.first << "}\"=\"" << entry.second
                << "\"";
      // Set the option in the ProviderOptions
      ret[option_name] = option_value;
    }
  }
  return ret;
}
std::string get_config_json_str(const onnxruntime::ProviderOptions& options1) {
  auto options = get_provider_option_from_env_variable(options1);
  try {
    auto data = morphizen::get_config_json(options);
    auto ret = std::string();
    auto status = google::protobuf::util::MessageToJsonString(
        data, &ret, google::protobuf::util::JsonPrintOptions());
    if (!status.ok()) {
      std::string err_msg =
          std::string{"failed to convert config to json string: "} + ret;
      err_msg += "\n" + status.ToString();
      LOG(FATAL) << err_msg;
    }
    return ret;
  } catch (const std::exception& e) {
    LOG(FATAL) << "Error: " << e.what() << std::endl;
    return "";
  }
}

Ort::SessionOptions*
get_session_option(const onnxruntime::ProviderOptions& options) {
  auto iter = options.find("session_options");
  if (iter == options.end()) {
    return nullptr;
  }
  return reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(iter->second));
}
} // namespace morphizen
