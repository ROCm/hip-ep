/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "./config.hpp"
#include <filesystem>
#include <onnxruntime_cxx_api.h>
namespace test_onnx_runner {
std::unique_ptr<Ort::Session>
create_session(const std::filesystem::path& model_path, Ort::Env& env,
               Ort::SessionOptions& session_options, const Config& config);

} // namespace test_onnx_runner
