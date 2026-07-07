/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "e2e_config_proto/test_case_config.pb.h"
#include <filesystem>
#include <memory>
#include <onnxruntime_c_api.h> // only for OrtLoggingLevel
#include <optional>
#include <unordered_map>

class E2ETestConfig {
public:
  static std::vector<std::unique_ptr<E2ETestConfig>>
  create(const std::filesystem::path& config_path);

public:
  const E2ETestConfigProto& proto() const { return config_proto_; }

public:
  explicit E2ETestConfig(const E2ETestConfigProto& proto);
  E2ETestConfig(const E2ETestConfig&) = delete;
  ~E2ETestConfig() = default;

private:
  E2ETestConfigProto config_proto_;
};
