/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "e2e_config_proto/test_case_config.pb.h"
#include <memory>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace morphizen_e2e_test {

class E2ETestSession {
public:
  explicit E2ETestSession(Ort::Env &env, Ort::SessionOptions &session_options,
                          const E2ETestSessionProto &session_proto);

  ~E2ETestSession() = default;
  E2ETestSession(const E2ETestSession &) = delete;
  E2ETestSession &operator=(const E2ETestSession &) = delete;

  void run();

private:
  const E2ETestSessionProto &session_proto_;
  Ort::Env &env_;
  Ort::SessionOptions &session_options_;

  std::vector<std::unique_ptr<Ort::Session>> ort_sessions_;
};

} // namespace morphizen_e2e_test
