/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

/**
 * @file morphizen-utils.hpp
 * @brief Main header for MorphiZen utility library
 *
 * This header provides convenient access to all utility components:
 * - Environment configuration with type safety
 * - Weak reference patterns (singleton and store)
 * - String parsing utilities
 *
 * Usage:
 * @code
 * #include <morphizen-utils/morphizen-utils.hpp>
 *
 * // Define environment parameter
 * DEF_ENV_PARAM(DEBUG_LEVEL, "0");
 *
 * // Use it
 * int level = ENV_PARAM(DEBUG_LEVEL);
 *
 * // Use weak singleton
 * auto instance = morphizen::utils::WeakSingleton<MyClass>::create();
 * @endcode
 */

#include "./env_config.hpp"
#include "./parse_value.hpp"
#include "./weak_refs.hpp"

/**
 * @namespace morphizen::utils
 * @brief Utility functions and templates for MorphiZen project
 *
 * This namespace contains general-purpose utilities that can be used
 * across different components of the MorphiZen project:
 *
 * - **Environment Configuration**: Type-safe environment variable access
 * - **Weak References**: Memory-safe singleton and object store patterns
 * - **String Parsing**: Robust parsing utilities with error checking
 */
namespace morphizen::utils {
// All functionality is defined in individual headers
}
