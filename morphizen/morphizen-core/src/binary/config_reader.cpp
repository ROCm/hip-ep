/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/plugin.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <optional>
#include <sstream>
#include <string>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_CONFIG_READER, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_CONFIG_READER) >= n)
DEF_ENV_PARAM_2(MORPHIZEN_CONFIG_PROVIDER_BACKEND, "onnxruntime_morphizen_ep",
                std::string)

namespace morphizen {

namespace config_default {
#include "config_json_binary.hpp"
}

static const char *get_default_config() {
  // `with_default_morphizen_config` and `config` are generated
  // automatically by
  // ${CMAKE_CURRENT_SOURCE_DIR}/src/binary/config_json_binary.hpp.py
  if (config_default::with_default_morphizen_config) {
    return (const char *)&config_default::config[0];
  }
  return nullptr;
}

static void JsonFileToMessage(const std::string &file_path,
                              google::protobuf::Message *message) {
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

static std::unique_ptr<google::protobuf::Struct>
get_protobuf_struct_from_config_file(const std::string &filename) {
  std::ifstream f(filename);
  // parse the json file into Struct message
  auto config = std::make_unique<google::protobuf::Struct>();
  JsonFileToMessage(filename, config.get());
  return config;
}

static google::protobuf::Struct
get_config_json(const onnxruntime::ProviderOptions &options) {
  google::protobuf::Struct ret;
  // update_log_level(options);
  auto morphizen_get_default_config_plugin =
      ::morphizen::Plugin::get(ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND));
  const char *default_config = get_default_config();
  if (default_config == nullptr) {
    if (morphizen_get_default_config_plugin) {
      MY_LOG(1) << "found plugin: "
                << ENV_PARAM(MORPHIZEN_CONFIG_PROVIDER_BACKEND);
      auto morphizen_get_default_config =
          morphizen_get_default_config_plugin->get_method<const char *>(
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
  return ret;
}

std::string get_config_json_str(const onnxruntime::ProviderOptions &options) {
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
  } catch (const std::exception &e) {
    LOG(FATAL) << "Error: " << e.what() << std::endl;
    return "";
  }
}
} // namespace morphizen
