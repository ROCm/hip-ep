/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "morphizen/morphizen.hpp"
#include "onnxruntime_api.hpp"
#include <string>
namespace morphizen {
/**
 * MorphiZen EP configuration reader.
 *
 * Loads ConfigProto-shaped JSON from the "config_file" provider option, or
 * from the embedded/plugin default when that option is absent.
 */
std::string get_config_json_str(const onnxruntime::ProviderOptions &options);
} // namespace morphizen
