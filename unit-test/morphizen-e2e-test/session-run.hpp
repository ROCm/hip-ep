/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./config.hpp"
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
namespace morphizen_e2e_test {
void run_session(Ort::Session& session,
                 const E2ETestSessionRunProto& run_proto);
} // namespace morphizen_e2e_test
