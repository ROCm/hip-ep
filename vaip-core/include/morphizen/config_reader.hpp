/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "onnxruntime_api.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
namespace onnxruntime {
using ProviderOptions = std::unordered_map<std::string, std::string>;
}
namespace vaip_core {
/**
 * Precedence: "config_file" option > "xclbin" option > XLNX_VART_FIRMWARE
 * environment variable At least one of above must be provided.
 *
 * Please note XLNX_VART_FIRMWARE can be set as a directory. But this would
 * leads to ambiguity. So, at VAIP, this would be treated as not set!
 *
 * Graph engine could find xclbin without the user setting XLNX_VART_FIRMWARE.
 * But, VAIP could not determine which one is actually used. So there are a few
 * possible conflicts: 1 User set the XLNX_VART_FIRMWARE as a directory and VAIP
 * couldn't find the any config. 2 Graph engine automatically found a different
 * xclbin file which is conflicting with xclbin and config_file option.
 *
 */
std::string get_config_json_str(const onnxruntime::ProviderOptions& options);
Ort::SessionOptions*
get_session_option(const onnxruntime::ProviderOptions& options);
} // namespace vaip_core
