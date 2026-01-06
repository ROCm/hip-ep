/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vaip/vaip_gsl.h>
#include <vector>
namespace vaip_core {
std::vector<char> get_mem_binary(const std::string& filename);
bool has_mem_binary(const std::string& filename);
std::optional<gsl::span<const char>>
get_mem_binary_span(const std::string& filename);

} // namespace vaip_core
