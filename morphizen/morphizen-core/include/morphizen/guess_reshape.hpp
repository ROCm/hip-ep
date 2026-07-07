/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include <cstdlib>
#include <morphizen/export.h>
#include <vector>
namespace morphizen {
MORPHIZEN_DLL_SPEC
std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
guess_reshape(const std::vector<int64_t> &shape_1,
              const std::vector<int64_t> &shape_2);
} // namespace morphizen
