/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen/morphizen.hpp"
#include "onnxruntime_api.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
namespace morphizen {
/**
 * MorphiZen EP configuration reader.
 *
 * Loads configuration from provider options, with "config_file" option
 * taking precedence over embedded defaults.
 */
std::string get_config_json_str(const onnxruntime::ProviderOptions& options);
Ort::SessionOptions*
get_session_option(const onnxruntime::ProviderOptions& options);
} // namespace morphizen
