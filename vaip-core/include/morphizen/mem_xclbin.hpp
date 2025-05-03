/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <string>
#include <vector>
namespace vaip_core {
std::vector<char> get_mem_xclbin(const std::string& filename);
bool has_mem_xclbin(const std::string& filename);
} // namespace vaip_core
