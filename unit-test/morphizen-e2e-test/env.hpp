/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "e2e_config_proto/test_case_config.pb.h"
#include "session-options.hpp"
#include <memory>

namespace morphizen_e2e_test {

class E2ETestEnv {
public:
  explicit E2ETestEnv(const E2ETestEnvProto& env_proto);
  ~E2ETestEnv();

  E2ETestEnv(const E2ETestEnv&) = delete;
  E2ETestEnv& operator=(const E2ETestEnv&) = delete;

  std::vector<std::unique_ptr<E2ETestSessionOptions>>
  create_e2e_test_session_options();

public:
  const E2ETestEnvProto& proto() const { return env_proto_; }
  Ort::Env& env() const { return *ort_env_; }
  const std::vector<Ort::ConstEpDevice>& selected_devices() const {
    return selected_devices_;
  }

private:
  const E2ETestEnvProto& env_proto_;
  std::unique_ptr<Ort::Env> ort_env_;
  std::vector<Ort::ConstEpDevice> selected_devices_;
};

} // namespace morphizen_e2e_test
