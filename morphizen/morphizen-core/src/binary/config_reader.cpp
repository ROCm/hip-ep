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
#include <morphizen-utils/json.hpp>
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

static json::Json parse_json_file(const std::string &file_path) {
  std::ifstream input(file_path);
  if (!input.is_open()) {
    std::string error_message = "Failed to open file: " + file_path;
    MY_LOG(1) << error_message;
    throw std::runtime_error(error_message);
  }

  std::string json_content((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  try {
    return json::parse(json_content);
  } catch (const json::ParseError &error) {
    std::string error_message = "Failed to parse JSON: " + std::string(error.what());
    MY_LOG(1) << error_message;
    throw std::runtime_error(error_message);
  }
}

static json::Json get_config_json(const onnxruntime::ProviderOptions &options) {
  json::Json ret = json::Json::object();
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
    ret = parse_json_file(opt_config_file.value().string());
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
    try {
      ret = json::parse(default_config);
    } catch (const json::ParseError &error) {
      std::string err_msg =
          std::string{"failed to parse default config: "} + default_config;
      err_msg += "\n";
      err_msg += error.what();
      LOG(FATAL) << err_msg;
    }
  }
  return ret;
}

std::string get_config_json_str(const onnxruntime::ProviderOptions &options) {
  try {
    auto data = morphizen::get_config_json(options);
    return json::dump(data);
  } catch (const std::exception &e) {
    LOG(FATAL) << "Error: " << e.what() << std::endl;
    return "";
  }
}
} // namespace morphizen
