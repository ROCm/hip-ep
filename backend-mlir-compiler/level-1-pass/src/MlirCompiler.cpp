/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"

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

// These are defined in pass_main.cpp
enum class ArtifactFormat { Native, LlvmIr };

struct CompilationConfig {
  ArtifactFormat artifactFormat;
  int optLevel;
  std::string outputFilename;
};

struct CompilationArtifact {
  std::string filename;
  std::vector<uint8_t> bytes;
  ArtifactFormat format;
};

namespace {

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

  // Check if symbol exists
  if (!plugin->has_method("morphizen_mlir_compile")) {
    LOG(ERROR) << "Symbol 'morphizen_mlir_compile' NOT found in DLL";
    return std::nullopt;
  }

  LOG(INFO) << "Calling morphizen_mlir_compile with JSON options: "
            << options_json;

  // Get method with explicit types (avoids template forwarding ref issues)
  // Signature: CompilerErrorCode (*)(const char*, const char*, const char*,
  // CompilerError*)
  auto func =
      plugin->get_method<CompilerErrorCode, const char *, const char *,
                         const char *, CompilerError *>(
          "morphizen_mlir_compile");

  if (func == nullptr) {
    LOG(ERROR) << "get_method returned nullptr for morphizen_mlir_compile";
    return std::nullopt;
  }

  // Call the function
  CompilerError error = {};
  auto result =
      func(mlir_bytecode.c_str(), temp_output_path.c_str(), options_json.c_str(), &error);

  if (result != COMPILER_SUCCESS) {
    LOG(ERROR) << "Compilation failed: " << error.message;
    return std::nullopt;
  }

  LOG(INFO) << "Compilation successful";
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

  // Build artifact
  CompilationArtifact artifact;
  artifact.filename = output_filename;
  artifact.bytes = std::move(buffer);
  artifact.format = config.artifactFormat;

  LOG(INFO) << "Artifact created: " << artifact.filename << " ("
            << artifact.bytes.size() << " bytes)";

  return artifact;
}

} // namespace level1pass
} // namespace hipdnn
