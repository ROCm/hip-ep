/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./session.hpp"
#include "e2e_config_proto/test_case_config.pb.h"
#include <memory>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>

namespace morphizen_e2e_test {

class E2ETestSessionOptions {
public:
  E2ETestSessionOptions(
      const E2ETestSessionOptionsProto &proto, Ort::Env &env,
      const std::vector<Ort::ConstEpDevice> &selected_devices);

  // delete copy constructor and assignment operator
  E2ETestSessionOptions(const E2ETestSessionOptions &) = delete;
  E2ETestSessionOptions &operator=(const E2ETestSessionOptions &) = delete;
  ~E2ETestSessionOptions() = default;

  std::vector<std::unique_ptr<E2ETestSession>> create_e2e_test_sessions();

  Ort::SessionOptions &get_ort_session_options() {
    return *ort_session_options_;
  }

public:
  const E2ETestSessionOptionsProto &proto() const { return proto_; }
  Ort::Env &env() const { return env_; }

private:
  const E2ETestSessionOptionsProto &proto_;
  Ort::Env &env_;

  std::unique_ptr<Ort::SessionOptions> ort_session_options_;
};

} // namespace morphizen_e2e_test
