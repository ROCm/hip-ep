/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "../../common/temp_path.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <fstream>
#include <glog/logging.h>
#include <sstream>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace hipdnn::level1pass {

namespace {

// Error codes from hip-compiler plugin C API
enum CompilerErrorCode {
  COMPILER_SUCCESS = 0,
  COMPILER_ERROR_INVALID_INPUT = 1,
  COMPILER_ERROR_PARSE_FAILED = 2,
  COMPILER_ERROR_PASS_FAILED = 3,
  COMPILER_ERROR_TRANSLATION_FAILED = 4,
  COMPILER_ERROR_LINKING_FAILED = 5,
  COMPILER_ERROR_IO_ERROR = 6,
  COMPILER_ERROR_COMPILATION_FAILED = 7,
  COMPILER_ERROR_VALIDATION_FAILED = 8,
  COMPILER_ERROR_INTERNAL = 99
};

// Error struct from hip-compiler plugin C API
struct CompilerError {
  CompilerErrorCode code;
  char message[1024];
};

// Build JSON options string from compilation config
std::string build_compiler_options_json(const CompilationConfig &config) {
  std::ostringstream json;
  json << "{";
  json << "\"opt_level\": " << config.optLevel;
  json << ", \"output_mode\": \"DLL\"";
  json << ", \"skip_constant_data\": "
       << (config.skipConstantData ? "true" : "false");
  json << "}";
  return json.str();
}

} // anonymous namespace

CompilationResult
MlirCompiler::compileFromBytecode(const std::string &mlir_bytecode,
                                  const CompilationConfig &config,
                                  morphizen::FileSystem *fs) {

  LOG(INFO) << "Compiling MLIR bytecode using hip-compiler plugin";
  LOG(INFO) << "Bytecode size: " << mlir_bytecode.size() << " bytes";

  CompilationResult outcome;

  auto plugin = morphizen::Plugin::get("hip-compiler");
  if (!plugin) {
    outcome.error_message = "Failed to load hip-compiler plugin";
    LOG(ERROR) << outcome.error_message;
    return outcome;
  }

  auto version = plugin->invoke<const char *>("hip_get_version");
  LOG(INFO) << "Plugin version: " << version;

  // No extension: compiler derives output format from output_mode in options
  // JSON
  std::string temp_output_path = mlir_compiler_utils::generateTempPath("");

  std::string options_json = build_compiler_options_json(config);

  LOG(INFO) << "Compilation options (JSON): " << options_json;

  if (!plugin->has_method("hip_compile_with_fs")) {
    outcome.error_message =
        "Symbol 'hip_compile_with_fs' NOT found in hip-compiler plugin";
    LOG(ERROR) << outcome.error_message;
    return outcome;
  }

  LOG(INFO) << "Calling hip_compile_with_fs with JSON options: "
            << options_json;

  // Signature: CompilerErrorCode (*)(const void*, size_t, const char*, const
  // char*, CompilerError*, void* fs)
  auto func =
      plugin->get_method<CompilerErrorCode, const void *, size_t, const char *,
                         const char *, CompilerError *, void *>(
          "hip_compile_with_fs");

  MY_LOG(2) << "get_method returned func = " << (void *)func;
  MY_LOG(2) << "Bytecode data() = " << (void *)mlir_bytecode.data();
  MY_LOG(2) << "Bytecode size() = " << mlir_bytecode.size();

  if (func == nullptr) {
    outcome.error_message =
        "get_method returned nullptr for hip_compile_with_fs";
    LOG(ERROR) << outcome.error_message;
    return outcome;
  }

  MY_LOG(2) << "About to call func with size = " << mlir_bytecode.size();

  CompilerError error = {};
  auto result =
      func(mlir_bytecode.data(), mlir_bytecode.size(), temp_output_path.c_str(),
           options_json.c_str(), &error, fs);

  if (result != COMPILER_SUCCESS) {
    // Capture plugin diagnostic (e.g. MLIR pass failure, SSA dominance
    // violation) so pass_main.cpp can surface it instead of silently letting
    // ORT route the subgraph elsewhere.
    outcome.error_message = std::string("hip-compiler plugin failed (code=") +
                            std::to_string(static_cast<int>(result)) +
                            "): " + error.message;
    LOG(ERROR) << outcome.error_message;
    return outcome;
  }

  LOG(INFO) << "Compilation successful";
  LOG(INFO) << "Reading artifact from: " << temp_output_path;

  std::ifstream file(temp_output_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    outcome.error_message =
        "Failed to open compiled artifact: " + temp_output_path;
    LOG(ERROR) << outcome.error_message;
    return outcome;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    outcome.error_message = "Failed to read compiled artifact bytes";
    LOG(ERROR) << outcome.error_message;
    file.close();
    return outcome;
  }
  file.close();

  std::remove(temp_output_path.c_str());

  CompilationArtifact artifact;
  artifact.filename = "model_compiled";
  artifact.bytes = std::move(buffer);
  artifact.format = config.artifactFormat;

  LOG(INFO) << "Artifact created: " << artifact.bytes.size() << " bytes";

  outcome.artifact = std::move(artifact);
  return outcome;
}

} // namespace hipdnn::level1pass
