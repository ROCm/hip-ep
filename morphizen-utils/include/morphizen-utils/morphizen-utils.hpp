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
 * - Environment configuration with type safety (re-exported from foundation)
 * - Weak reference patterns (singleton and store)
 * - String parsing utilities (re-exported from foundation)
 * - Plugin loading system
 * - Cleanup utilities
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

// Import foundation utilities (generic utilities moved to morphizen-foundation)
#include <morphizen-foundation/env_config.hpp>
#include <morphizen-foundation/parse_value.hpp>

// Local utilities (MorphiZen-specific)
#include "./cleanup.hpp"
#include "./morphizen_plugin.hpp"
#include "./weak_refs.hpp"

/**
 * @namespace morphizen::utils
 * @brief Utility functions and templates for MorphiZen project
 *
 * This namespace contains framework-specific utilities for the MorphiZen
 * project:
 *
 * - **Environment Configuration**: Type-safe environment variable access (from
 * foundation)
 * - **String Parsing**: Robust parsing utilities (from foundation)
 * - **Weak References**: Memory-safe singleton and object store patterns
 * - **Plugin System**: Dynamic plugin loading
 * - **Cleanup Utilities**: Framework cleanup helpers
 */
namespace morphizen::utils {
// Re-export foundation namespace for backwards compatibility
using namespace morphizen::foundation;
} // namespace morphizen::utils
