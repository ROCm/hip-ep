/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"
#include "CompilationArtifact.h"
#include "CompilationConfig.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <sstream>

using namespace mlir_compiler_local;

namespace hipdnn {
namespace level1pass {

namespace {

// Call morphizen_mlir_compile with JSON-based options
//
// NEW API (morphizen-mlir-compiler integration.morphizen-mlir-compiler branch):
// - Function: morphizen_mlir_compile(input_mlir, output_path, options_json, error)
// - Replaces old morphizen_mlir_compile_bytecode with 5 parameters
// - Uses JSON string for options instead of C struct (extensible, no ABI coupling)
CompilerErrorCode
call_compile_with_json(morphizen::Plugin *plugin, const std::string &input_mlir,
                       const char *output_path, const std::string &options_json,
                       CompilerError *error) {
  // Check if symbol exists
  if (!plugin->has_method("morphizen_mlir_compile")) {
    LOG(ERROR) << "Symbol 'morphizen_mlir_compile' NOT found in DLL";
    return COMPILER_ERROR_INTERNAL;
  }

  LOG(INFO) << "Calling morphizen_mlir_compile with JSON options: "
            << options_json;

  // Get method with explicit types (avoids template forwarding ref issues)
  // Signature: CompilerErrorCode (*)(const char*, const char*, const char*, CompilerError*)
  auto func =
      plugin->get_method<CompilerErrorCode, const char *, const char *,
                         const char *, CompilerError *>(
          "morphizen_mlir_compile");

  if (func == nullptr) {
    LOG(ERROR) << "get_method returned nullptr for morphizen_mlir_compile";
    return COMPILER_ERROR_INTERNAL;
  }

  // Call the function
  return func(input_mlir.c_str(), output_path, options_json.c_str(), error);
}

// Generate safe temporary filename (replaces deprecated std::tmpnam)
std::string generateTempPath() {
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();

  static int counter = 0;
  std::string filename = "morphizen_mlir_" + std::to_string(timestamp) + "_" +
                         std::to_string(counter++) + ".dll";

#ifdef _WIN32
  return filename; // Current directory on Windows
#else
  return std::string("/tmp/") + filename;
#endif
}

} // anonymous namespace

std::optional<CompilationArtifact>
MlirCompiler::compileFromBytecode(const std::string &mlir_bytecode,
                                  const CompilationConfig &config) {

  LOG(INFO) << "Compiling MLIR bytecode using morphizen-mlir-compiler plugin";
  LOG(INFO) << "Bytecode size: " << mlir_bytecode.size() << " bytes";

  // Load plugin via MorphiZen Plugin API
  auto plugin = morphizen::Plugin::get("morphizen-mlir-compiler");
  if (!plugin) {
    LOG(ERROR) << "Failed to load morphizen-mlir-compiler plugin";
    return std::nullopt;
  }

  // Get plugin version
  auto version = plugin->invoke<const char *>("morphizen_mlir_get_version");
  LOG(INFO) << "Plugin version: " << version;

  // Generate output filename
  std::string output_filename = config.outputFilename.empty()
                                    ? "model_compiled.dll"
                                    : config.outputFilename;

  // Generate temporary output path
  std::string temp_output_path = generateTempPath();

  // Build JSON options string from config
  // Format: {"opt_level": N, "output_mode": "OUTPUT_MODE_DLL"}
  std::ostringstream json;
  json << "{";
  json << "\"opt_level\": " << config.optLevel;
  json << ", \"output_mode\": \"OUTPUT_MODE_DLL\"";
  json << "}";
  std::string options_json = json.str();

  LOG(INFO) << "Compilation options (JSON): " << options_json;

  // Measure compilation time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Call new JSON-based C API
  CompilerError error = {};
  auto result = call_compile_with_json(
      plugin, mlir_bytecode, temp_output_path.c_str(), options_json, &error);

  auto end_time = std::chrono::high_resolution_clock::now();
  int64_t compilation_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count();

  if (result != COMPILER_SUCCESS) {
    LOG(ERROR) << "Compilation failed: " << error.message;
    return std::nullopt;
  }

  LOG(INFO) << "Compilation successful (" << compilation_ms << " ms)";
  LOG(INFO) << "Reading artifact from: " << temp_output_path;

  // Read compiled DLL into memory
  std::ifstream file(temp_output_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open compiled artifact: " << temp_output_path;
    return std::nullopt;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    LOG(ERROR) << "Failed to read compiled artifact";
    file.close();
    return std::nullopt;
  }
  file.close();

  // Clean up temporary file
  std::remove(temp_output_path.c_str());

  // Build artifact with timing information
  CompilationArtifact artifact;
  artifact.filename = output_filename;
  artifact.bytes = std::move(buffer);
  artifact.format = config.artifactFormat;
  artifact.compilation_ms = compilation_ms; // Total compilation time
  artifact.linking_ms = 0;                  // Not separately measured yet

  LOG(INFO) << "Artifact created: " << artifact.filename << " ("
            << artifact.bytes.size() << " bytes)";

  return artifact;
}

} // namespace level1pass
} // namespace hipdnn
