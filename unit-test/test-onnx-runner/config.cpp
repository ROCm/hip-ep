/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./config.hpp"
#include <google/protobuf/message.h>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <filesystem>
#include <fstream>

std::unordered_map<std::string, std::unique_ptr<Config>>
Config::create(const std::filesystem::path& config_path) {

  // Read the JSON file into a string
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    throw std::runtime_error("Could not open config file: " +
                             std::filesystem::absolute(config_path).string());
  }
  std::string file_str((std::istreambuf_iterator<char>(config_file)),
                       std::istreambuf_iterator<char>());
  config_file.close();
  const std::string json_str_prefix = "{ \"configs\": ";
  const std::string json_str_suffix = "}";
  std::string json_str;
  json_str.reserve(json_str_prefix.size() + file_str.size() +
                   json_str_suffix.size());
  json_str.append(json_str_prefix);
  json_str.append(file_str.begin(), file_str.end());
  json_str.append(json_str_suffix);

  // Parse the JSON string into the protobuf message
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true; // Ignore unknown fields

  EnvConfigProto root;
  auto status =
      google::protobuf::util::JsonStringToMessage(json_str, &root, options);

  if (!status.ok()) {
    throw std::runtime_error("Failed to parse JSON: " + json_str +
                             " Status : " + status.ToString());
  }

  auto ret = std::unordered_map<std::string, std::unique_ptr<Config>>();
  for (const auto& [name, proto] : root.configs()) {
    ret[name] = std::make_unique<Config>(name, proto);
  }
  return ret;
}

Config::Config(const std::string& name, const OrtConfigProto& proto)
    : name_(name), config_proto_(std::move(proto)) {}

bool Config::enable_vitisai_ep() const {
  return config_proto_.has_enable_vitisai_ep()
             ? config_proto_.enable_vitisai_ep()
             : true;
}
bool Config::use_memory_model() const {
  return config_proto_.use_memory_model();
}
int Config::session_count() const {
  return config_proto_.has_session_count() ? config_proto_.session_count() : 1;
}
int Config::batch_number() const {
  return config_proto_.has_batch_number() ? config_proto_.batch_number() : 1;
}

std::optional<std::string>
Config::get_session_config_option(const std::string& key) const {
  auto it = config_proto_.session_options().session_configs().find(key);
  if (it != config_proto_.session_options().session_configs().end()) {
    return it->second;
  }
  return std::nullopt;
}
std::optional<std::string>
Config::get_provider_option(const std::string& key) const {
  auto it = config_proto_.session_options().provider_options().find(key);
  if (it != config_proto_.session_options().provider_options().end()) {
    return it->second;
  }
  return std::nullopt;
}

std::filesystem::path
Config::get_ctx_model_path(const std::filesystem::path& model_path) const {
  auto ctx_path_name =
      model_path.stem().string() + "_ctx" + model_path.extension().string();
  auto ctx_path = model_path.parent_path() / ctx_path_name;

  auto cache_context_file_path =
      get_session_config_option("ep.context_file_path");
  if (cache_context_file_path.has_value()) {
    ctx_path = std::filesystem::u8path(cache_context_file_path.value());
  }
  return ctx_path;
}

bool Config::embed_mode() const {
  auto ret = false;
  auto is_embed_mode = get_provider_option("ep.context_embed_mode");
  if (is_embed_mode.has_value()) {
    ret = is_embed_mode.value() != "0";
  }
  return ret;
}
bool Config::ep_context_enable() const {
  auto ret = false;
  auto ep_context_enable = get_session_config_option("ep.context_enable");
  if (ep_context_enable.has_value()) {
    ret = ep_context_enable.value() != "0";
  }
  return ret;
}

OrtLoggingLevel Config::ort_log_level() const {
  auto log_level = get_provider_option("log_level");
  if (log_level.has_value()) {
    if (log_level.value() == "verbose") {
      return ORT_LOGGING_LEVEL_VERBOSE;
    } else if (log_level.value() == "info") {
      return ORT_LOGGING_LEVEL_INFO;
    } else if (log_level.value() == "warning") {
      return ORT_LOGGING_LEVEL_WARNING;
    } else if (log_level.value() == "error") {
      return ORT_LOGGING_LEVEL_ERROR;
    } else if (log_level.value() == "fatal") {
      return ORT_LOGGING_LEVEL_FATAL;
    }
  }
  return ORT_LOGGING_LEVEL_WARNING;
}

std::string Config::ort_log_id() const { return "test_onnx_runner"; }

void Config::set_session_config_option(const std::string& key,
                                       const std::string& value) {
  auto& session_configs =
      *config_proto_.mutable_session_options()->mutable_session_configs();
  session_configs[key] = value;
}
