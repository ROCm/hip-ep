/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <stdint.h>
#include <string>

namespace morphizen {
const std::string get_lib_name();
const std::string get_lib_id();
uint32_t get_morphizen_version_major();
uint32_t get_morphizen_version_minor();
uint32_t get_morphizen_version_patch();

extern "C" uint32_t morphizen_get_version();

// Additional version info from DLL resource (matching version.rc.in)
// These functions read the version information from the DLL's embedded resource
const std::string get_dll_company_name();
const std::string get_dll_product_name();
const std::string get_dll_legal_copyright();
const std::string get_dll_file_version();
const std::string get_dll_product_version();
const std::string get_dll_file_description();

} // namespace morphizen
extern "C" const char *morphizen_get_build_info();
