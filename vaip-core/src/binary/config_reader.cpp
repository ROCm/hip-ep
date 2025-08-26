/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip_plugin.hpp"
#include "vaip/vaip_ort_api.h"
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
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
DEF_ENV_PARAM_2(DEBUG_LOG_LEVEL, "error", std::string)
DEF_ENV_PARAM(XLNX_ENABLE_BATCH, "0")
DEF_ENV_PARAM(NUM_OF_DPU_RUNNERS, "1")
DEF_ENV_PARAM_2(VAIP_CONFIG_PROVIDER_BACKEND, "onnxruntime_vitisai_ep",
                std::string)
// this is actually provider options, for backward compatibility, we keep
// this key in the root of the json.
static constexpr char kProviderOptions[] = "sessionOptions";
static constexpr char kSessionConfig[] = "ort_session_config";
static constexpr char kSessionOptionPtr[] = "session_options";
static constexpr char kEpProviderOptionPrefix[] =
    "ep.vitisaiexecutionprovider.";
namespace vaip_core {
static const char* get_default_config() {
#include "config_json_binary.hpp"
  // `with_default_vaip_config` and `config` are generated
  // automatically by
  // ${CMAKE_CURRENT_SOURCE_DIR}/src/xclbin/config_json_binary.hpp.py
  if (with_default_vaip_config) {
    return (const char*)&config[0];
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

static void
update_enable_batch(const onnxruntime::ProviderOptions& session_option) {
  if (session_option.find("xlnx_enable_batch") != session_option.end()) {
    std::string enable_batch = session_option.at("xlnx_enable_batch");
    if (enable_batch == "1") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    } else if (enable_batch == "0") {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 0;
    } else {
      ENV_PARAM(XLNX_ENABLE_BATCH) = 1;
    }
  }
}

static void
update_num_dpu_runners(const onnxruntime::ProviderOptions& session_option) {
  int num_of_dpu_runners = 1;
  if (session_option.find("num_of_dpu_runners") != session_option.end()) {
    std::string str_of_dpu_runners = session_option.at("num_of_dpu_runners");
    num_of_dpu_runners = atoi(str_of_dpu_runners.c_str());
    if (num_of_dpu_runners <= 0 || num_of_dpu_runners > 8) {
      return;
    }
    ENV_PARAM(NUM_OF_DPU_RUNNERS) = num_of_dpu_runners;
#ifdef _WIN32
    _putenv_s("NUM_OF_DPU_RUNNERS", std::to_string(num_of_dpu_runners).c_str());
#else
    setenv("NUM_OF_DPU_RUNNERS", std::to_string(num_of_dpu_runners).c_str(), 1);
#endif
  }
}

static void
update_log_level(const onnxruntime::ProviderOptions& session_option) {
  std::string log_level = ENV_PARAM(DEBUG_LOG_LEVEL);
  if (session_option.find("log_level") != session_option.end()) {
    log_level = session_option.at("log_level");
  }

  if (log_level == "info") {
    FLAGS_minloglevel = google::GLOG_INFO;
  } else if (log_level == "warning") {
    FLAGS_minloglevel = google::GLOG_WARNING;
  } else if (log_level == "error") {
    FLAGS_minloglevel = google::GLOG_ERROR;
  } else if (log_level == "fatal") {
    FLAGS_minloglevel = google::GLOG_FATAL;
  } else {
    FLAGS_minloglevel = google::GLOG_ERROR;
  }
}
static void set_session_config(google::protobuf::Struct& ret,
                               const std::string& key,
                               const std::string& value) {
  if (key.rfind(kEpProviderOptionPrefix, 0) == 0) {
    auto key2 = key.substr(sizeof(kEpProviderOptionPrefix) - 1);
    MY_LOG(1) << "convert " << key << " to " << key2 << " and set "
              << "provider_options"
              << "[" << key2 << "]=\"" << value << "\"";
    // The key is prefixed with "ep.vitisaiexecutionprovider."
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
#if VAIP_ORT_API_MAJOR >= 10
  auto options = reinterpret_cast<Ort::SessionOptions*>(
      (uintptr_t)std::stoull(entry_second));
  VAIP_ORT_API(session_option_configuration)
  (&session_config_options_entry_list, options,
   [](void* mmap, const char* name, const char* value) {
     auto* map_ptr =
         reinterpret_cast<std::map<std::string, std::string>*>(mmap);
     map_ptr->insert({name, value});
   });
#else
  LOG(WARNING) << "ORT API version is less than 10, now used is "
               << VAIP_ORT_API_MAJOR
               << ", session options will not be restored.";
#endif
  for (const auto& option : session_config_options_entry_list) {
    set_session_config(ret, option.first, option.second);
  }
}

static google::protobuf::Struct
get_config_json(const onnxruntime::ProviderOptions& options) {
  google::protobuf::Struct ret;
  update_log_level(options);
  update_enable_batch(options);
  update_num_dpu_runners(options);
  auto vaip_get_default_config_plugin =
      ::vaip_core::Plugin::get(ENV_PARAM(VAIP_CONFIG_PROVIDER_BACKEND));
  const char* default_config = get_default_config();
  if (default_config == nullptr) {
    if (vaip_get_default_config_plugin) {
      MY_LOG(1) << "found plugin: " << ENV_PARAM(VAIP_CONFIG_PROVIDER_BACKEND);
      auto vaip_get_default_config =
          vaip_get_default_config_plugin->get_method<const char*>(
              "vaip_get_default_config");
      if (vaip_get_default_config) {
        MY_LOG(1) << "found symbol: vaip_get_default_config from "
                  << ENV_PARAM(VAIP_CONFIG_PROVIDER_BACKEND);
        default_config = vaip_get_default_config();
      } else {
        MY_LOG(1) << "cannot found symbol: vaip_get_default_config from "
                  << ENV_PARAM(VAIP_CONFIG_PROVIDER_BACKEND);
      }
    } else {
      MY_LOG(1) << "cannot found plugin: "
                << ENV_PARAM(VAIP_CONFIG_PROVIDER_BACKEND)
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
      LOG(FATAL) << "no default vaip_config.json, "
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
  auto xclbin_in_config_file =
      ret.fields().find("xclbin") != ret.fields().end();
  const std::string ort_session_config_prefix =
      std::string(kSessionConfig) + ".";

  for (const auto& entry : options) {
    MY_LOG(1) << "process provider_option[\"" << entry.first << "\"]= \""
              << entry.second << "\"";
    if (entry.first == "xclbin" && xclbin_in_config_file) {
      // FIXME: below comments might not be accurate.
      //
      // In the case where xclbin are (mistakenly) specified both in
      // `ProviderOptions["config_file"]` and in `ProvidersOptions["xclbin"]`.
    } else if (entry.first == kSessionOptionPtr) {
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
  // with "VITISAI_EP_PROVIER_OPTION."
  const std::string prefix = "VITISAI_EP_PROVIDER_OPTION_";
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
    auto data = vaip_core::get_config_json(options);
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
} // namespace vaip_core
