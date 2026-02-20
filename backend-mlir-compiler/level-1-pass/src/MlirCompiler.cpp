/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"
#include "CompilationArtifact.h"
#include "CompilationConfig.h"

// C API types (no linking - DLL loaded at runtime via Plugin API)
#include "morphizen-mlir-compiler/compiler_api.h"
#include "morphizen-mlir-compiler/compiler_types.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "morphizen/morphizen.hpp"
#include "morphizen/plugin.hpp"
#include <chrono>
#include <fstream>
#include <glog/logging.h>

namespace hipdnn {
namespace level1pass {

namespace {

// WORKAROUND: Bypass template forwarding reference issue in morphizen
// Plugin::invoke
//
// Problem: Plugin::invoke<R>(name, args...) uses template parameter Args&&...
// (forwarding refs), which causes get_method<R, Args&&...> to create function
// pointer typedef with rvalue refs:
//   e.g., CompilerErrorCode (*)(const void*&&, size_t&&, ...)
// This never matches C function signatures, causing symbol lookup to fail.
//
// Solution: Use get_method with EXPLICIT VALUE TYPES (not forwarding refs) to
// match C signature.
//
// TODO: Fix morphizen_plugin.hpp::get_method to use std::decay<Args>::type...
// in typedef
CompilerErrorCode
call_compile_bytecode_direct(morphizen::Plugin *plugin, const void *bytecode,
                             size_t bytecode_size, const char *output_path,
                             const CompilationOptions *options,
                             CompilerError *error) {
  // First check if symbol exists (without template type checking)
  if (!plugin->has_method("morphizen_mlir_compile_bytecode")) {
    LOG(ERROR) << "Symbol 'morphizen_mlir_compile_bytecode' NOT found in DLL "
                  "via has_method";
    return COMPILER_ERROR_INTERNAL;
  }

  LOG(INFO) << "Symbol 'morphizen_mlir_compile_bytecode' exists in DLL, "
               "getting typed function pointer...";

  // Call get_method with EXPLICIT types (not auto-deduced with &&)
  // This creates: CompilerErrorCode (*)(const void*, size_t, const char*, const
  // CompilationOptions*, CompilerError*)
  auto func =
      plugin->get_method<CompilerErrorCode, const void *, size_t, const char *,
                         const CompilationOptions *, CompilerError *>(
          "morphizen_mlir_compile_bytecode");

  if (func == nullptr) {
    LOG(ERROR) << "get_method returned nullptr despite has_method=true - TYPE "
                  "MISMATCH in template";
    LOG(ERROR) << "Expected: CompilerErrorCode (*)(const void*, size_t, const "
                  "char*, const CompilationOptions*, CompilerError*)";
    return COMPILER_ERROR_INTERNAL;
  }

  LOG(INFO) << "Successfully got typed function pointer, calling...";

  // Call the function with exact argument types
  return func(bytecode, bytecode_size, output_path, options, error);
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

  // Setup compilation options
  CompilationOptions opts = {}; // Zero-initialize to prevent garbage values
  plugin->invoke<void>("morphizen_mlir_get_default_options", &opts);

  opts.from_onnx_mlir = 1; // Input is ONNX MLIR dialect
  opts.opt_level = config.optLevel;
  opts.output_mode = OUTPUT_MODE_DLL;
  opts.mock_runtime = config.useMockRuntime ? 1 : 0;
  opts.memory_alignment = 64;
  opts.verbose = 0;
  opts.keep_intermediates = 0;

  // Measure compilation time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Call C API via Plugin (using direct wrapper to avoid template forwarding
  // ref issue)
  CompilerError error;
  auto result = call_compile_bytecode_direct(
      plugin, mlir_bytecode.data(), mlir_bytecode.size(),
      temp_output_path.c_str(), &opts, &error);

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
