/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

namespace morphizen {
/**
 * @brief Option key for specifying the target name.
 *
 * There are multiply compiler backends supported by MorphiZen EP, some of them
 * could work together to support a model. A set of predefined `TargetProto`
 * define how to compose these compiler backends, `target` is used to select
 * which `TargetProto` is in use.
 */
static constexpr const char *kProviderOptionTarget = "target";
/**
 * @brief Option key for specifying the configuration file path.
 *
 * This constant can be used to set or retrieve the path to a configuration file
 * that contains additional settings for the provider.
 */
static constexpr const char *kProviderOptionConfigFile = "config_file";

/**
 * @brief Option key for specifying the log level in provider configurations.
 *
 * This constant can be used to set or retrieve the desired logging level
 * for a provider, such as "debug", "info", "warn", or "error".
 */
static constexpr const char *kProviderOptionLogLevel = "log_level";
/**
 * @brief Option key for specifying the cache directory in provider
 * configurations.
 *
 * This constant can be used to set or retrieve the directory path where
 * cached files should be stored by the provider.
 */
static constexpr const char *kProviderOptionCacheDir = "cache_dir";

/**
 * @brief Option key to enable TAR memory-mapped execution provider context.
 *
 * This constant string is used as a key for specifying whether we should
 * enable the TAR memory-mapped execution provider context in the system.
 */
static constexpr const char *kProviderOptionEpContextEnableMmap =
    "ep_context_enable_mmap";

} // namespace morphizen
