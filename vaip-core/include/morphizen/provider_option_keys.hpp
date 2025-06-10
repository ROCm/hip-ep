/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

namespace vaip_core {
/**
 * @brief Option key for specifying the log level in provider configurations.
 *
 * This constant can be used to set or retrieve the desired logging level
 * for a provider, such as "debug", "info", "warn", or "error".
 */
static constexpr const char* kProviderOptionLogLevel = "log_level";
/**
 * @brief Option key for specifying the cache directory in provider
 * configurations.
 *
 * This constant can be used to set or retrieve the directory path where
 * cached files should be stored by the provider.
 */
static constexpr const char* kProviderOptionCacheDir = "cache_dir";
} // namespace vaip_core
