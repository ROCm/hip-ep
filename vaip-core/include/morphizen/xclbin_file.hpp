/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/vaip.hpp"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
namespace vaip_core {
class PassContext;
VAIP_DLL_SPEC std::optional<uint64_t>
get_xclbin_fingerprint(const PassContext& pass_context,
                       const std::filesystem::path& filename);
} // namespace vaip_core
