/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "./pass_imp.hpp"
#include <filesystem>
#include <string>
namespace morphizen {
bool file_exists(const std::filesystem::path& filename);
// return a cache directory.
std::filesystem::path get_cache_file_name(const PassContext& context,
                                          const std::string& filename);
void update_cache_dir(PassContextImp& context);
} // namespace morphizen
