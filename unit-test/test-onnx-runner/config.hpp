/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "env_config_proto/env_config.pb.h"
#include <filesystem>
#include <memory>
#include <onnxruntime_c_api.h> // only for OrtLoggingLevel
#include <optional>
#include <unordered_map>

class Config {
public:
  static std::unordered_map<std::string, std::unique_ptr<Config>>
  create(const std::filesystem::path& config_path);

public:
  const OrtConfigProto& proto() const { return config_proto_; }
  const std::string& name() const { return name_; }
  bool enable_vitisai_ep() const;
  bool use_memory_model() const;
  int session_count() const;
  int batch_number() const;

  std::optional<std::string>
  get_session_config_option(const std::string& key) const;
  std::optional<std::string> get_provider_option(const std::string& key) const;

  OrtLoggingLevel ort_log_level() const;
  std::string ort_log_id() const;

  std::filesystem::path
  get_ctx_model_path(const std::filesystem::path& model_path) const;
  bool embed_mode() const;
  bool ep_context_enable() const;

  void set_session_config_option(const std::string& key,
                                 const std::string& value);

public:
  Config(const std::string& name, const OrtConfigProto& proto);
  Config(const Config&) = delete;
  ~Config() = default;

private:
  std::string name_;
  OrtConfigProto config_proto_;
};
