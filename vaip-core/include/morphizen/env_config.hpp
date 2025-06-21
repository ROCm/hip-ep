/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Use the new morphizen-utils library instead of local implementation
#include <morphizen-utils/morphizen-utils.hpp>

// Re-export the utilities in the morphizen namespace for backward compatibility
namespace morphizen {

// Re-export the template helpers with original names for compatibility
template <typename T>
using env_config_helper = ::morphizen::utils::env_config_helper<T>;

template <typename T, typename env_name>
using env_config = ::morphizen::utils::env_config<T, env_name>;

} // namespace morphizen

// The macros are already defined in morphizen-utils and work the same way
// DEF_ENV_PARAM, DEF_ENV_PARAM_2, and ENV_PARAM are available directly
