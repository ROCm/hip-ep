/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
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
static google::protobuf::util::Status
JsonFileToMessage(const std::string& file_path,
                  google::protobuf::Message* message) {
  std::ifstream input(file_path);
  if (!input.is_open()) {
    std::string error_message = "Failed to open file: " + file_path;
    MY_LOG(1) << error_message;
    return google::protobuf::util::Status(
        google::protobuf::util::StatusCode::kInvalidArgument, error_message);
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
    return google::protobuf::util::Status(
        google::protobuf::util::StatusCode::kInvalidArgument, error_message);
  }

  return status; // Return the successful status
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
  auto status = JsonFileToMessage(filename, config.get());
  if (!status.ok()) {
    std::string err_msg =
        std::string{"failed to parse config file: "} + filename;
    err_msg += "\n" + status.ToString();
    throw std::runtime_error(err_msg);
  }
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
    set_struct_value(ret, "ort_session_config", option.first, option.second);
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
  bool config_set = options.find("config_file") != options.end();
  if (config_set) {
    std::string config_file = options.at("config_file");
    MY_LOG(1) << " overwrite default config, read if from " << config_file;
    auto struct_from_config_file =
        get_protobuf_struct_from_config_file(config_file);
    if (struct_from_config_file == nullptr) {
      LOG(FATAL) << "failed to parse config file: " << config_file;
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
  for (const auto& entry : options) {
    // In the case where xclbin are (mistakenly) specified both in
    // `ProviderOptions["config_file"]` and in `ProvidersOptions["xclbin"]`.
    if (entry.first == "xclbin" && xclbin_in_config_file) {
      continue;
    }
    if (entry.first == "session_options") {
      // The key here is "session_options," and the value is a string that
      // points to the session_options object.
      restore_session_options(ret, entry.second);
      continue;
    }
    // this is actually provider options, for backward compatibility, we keep
    // this key in the root of the json.
    auto kProviderOptions = "sessionOptions";
    set_struct_value(ret, kProviderOptions, entry.first, entry.second);
  }
  return ret;
}

std::string get_config_json_str(const onnxruntime::ProviderOptions& options) {
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
