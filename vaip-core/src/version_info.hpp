/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <stdint.h>
#include <string>

namespace vaip_core {
const std::string get_lib_name();
const std::string get_lib_id();
uint32_t get_vaip_version_major();
uint32_t get_vaip_version_minor();
uint32_t get_vaip_version_patch();
extern "C" uint32_t vaip_get_version();

} // namespace vaip_core
extern "C" const char* morphizen_get_build_info();
