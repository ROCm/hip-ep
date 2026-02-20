/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "CompilationConfig.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// Environment parameters (must be at global scope before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace hipdnn {
namespace level1pass {

CompilationConfig CompilationConfig::fromProviderOptions(
    const std::shared_ptr<morphizen::PassContext> &context) {
  CompilationConfig config = defaultConfig();

  try {
    // Parse artifact format
    std::string artifact_format_str =
        context->get_provider_option("artifact_format", "native");
    if (artifact_format_str == "llvm_ir") {
      config.artifactFormat = ArtifactFormat::LlvmIr;
    } else if (artifact_format_str != "native") {
      MY_LOG(1) << "Unknown artifact_format: " << artifact_format_str
                << ", using default: native";
    }

    // Parse optimization level
    std::string opt_level_str =
        context->get_provider_option("optimization_level", "2");
    config.optLevel = std::stoi(opt_level_str);

    // Parse output filename
    config.outputFilename = context->get_provider_option("output_filename", "");

    // Parse mock runtime flag
    std::string use_mock_runtime_str =
        context->get_provider_option("use_mock_runtime", "true");
    config.useMockRuntime =
        (use_mock_runtime_str == "true" || use_mock_runtime_str == "1");

  } catch (const std::exception &ex) {
    MY_LOG(1) << "Failed to parse provider options: " << ex.what()
              << ", using defaults";
  }

  return config;
}

CompilationConfig CompilationConfig::defaultConfig() {
  CompilationConfig config;
  config.artifactFormat = ArtifactFormat::Native;
  config.optLevel = 2;
  config.outputFilename = "";
  config.useMockRuntime = true;
  return config;
}

} // namespace level1pass
} // namespace hipdnn
