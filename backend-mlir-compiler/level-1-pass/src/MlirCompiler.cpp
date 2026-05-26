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
#include <cstring>
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

// Mirrors `CompilationOutputs` in include/hip/compiler_types.h. Layout
// MUST stay in sync with that struct (binary-compatible across the plugin
// boundary). Pre-allocate the buffers in `compileFromBytecode` and pass
// the struct through to the compiler — replaces the prior two-call
// thread-local stash retrieval (`hip_get_last_compile_output_*`).
//
// Buffer sizing rationale: shapes_needed is bounded by
//   1 + num_outputs * (1 + max_rank);
// origins_needed by
//   1 + num_outputs * (1 + 3 * max_rank).
// Real models top out around tens of outputs × tens of dims; 4096 int64s
// (32 KB) per buffer is generous enough that we never expect truncation.
// We assert (`*_needed <= *_capacity`) and warn loudly if a model ever
// pushes past — that's the trigger to bump the constants.
struct PluginCompilationOutputs {
  int64_t *shapes_buf;
  int64_t shapes_capacity;
  int64_t shapes_needed;
  int64_t *origins_buf;
  int64_t origins_capacity;
  int64_t origins_needed;
};

static constexpr int64_t kInferOutputsBufferInt64s = 4096;

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

std::optional<CompilationArtifact>
MlirCompiler::compileFromBytecode(const std::string &mlir_bytecode,
                                  const CompilationConfig &config,
                                  morphizen::FileSystem *fs) {

  LOG(INFO) << "Compiling MLIR bytecode using hip-compiler plugin";
  LOG(INFO) << "Bytecode size: " << mlir_bytecode.size() << " bytes";

  // Load plugin via MorphiZen Plugin API
  auto plugin = morphizen::Plugin::get("hip-compiler");
  if (!plugin) {
    LOG(ERROR) << "Failed to load hip-compiler plugin";
    return std::nullopt;
  }

  // Get plugin version
  auto version = plugin->invoke<const char *>("hip_get_version");
  LOG(INFO) << "Plugin version: " << version;

  // Generate temporary output path for compilation
  // No extension: compiler derives output format from output_mode in options
  // JSON
  std::string temp_output_path = mlir_compiler_utils::generateTempPath("");

  // Build JSON options string from config
  std::string options_json = build_compiler_options_json(config);

  LOG(INFO) << "Compilation options (JSON): " << options_json;

  // Check if symbol exists
  if (!plugin->has_method("hip_compile_with_fs")) {
    LOG(ERROR) << "Symbol 'hip_compile_with_fs' NOT found in DLL";
    return std::nullopt;
  }

  LOG(INFO) << "Calling hip_compile_with_fs with JSON options: "
            << options_json;

  // Get method with explicit types (avoids template forwarding ref issues).
  // Signature: CompilerErrorCode (*)(const void*, size_t, const char*,
  //   const char*, CompilerError*, void* fs, PluginCompilationOutputs*)
  // The trailing struct pointer is the InferOnnxShapes-results out-channel
  // (replaces the prior thread-local stash + two-call discovery).
  auto func =
      plugin->get_method<CompilerErrorCode, const void *, size_t, const char *,
                         const char *, CompilerError *, void *,
                         PluginCompilationOutputs *>("hip_compile_with_fs");

  MY_LOG(2) << "get_method returned func = " << (void *)func;
  MY_LOG(2) << "Bytecode data() = " << (void *)mlir_bytecode.data();
  MY_LOG(2) << "Bytecode size() = " << mlir_bytecode.size();

  if (func == nullptr) {
    LOG(ERROR) << "get_method returned nullptr for hip_compile_with_fs";
    return std::nullopt;
  }

  // Pre-allocate output buffers. The compiler will write the full
  // packed-int64 layout (shapes + per-dim SSA origins) into these. If a
  // model ever pushes past the fixed capacity we warn loudly so the
  // constant can be bumped — bounded by graph signature size, not
  // anything content-dependent.
  std::vector<int64_t> shapes_storage(
      static_cast<size_t>(kInferOutputsBufferInt64s), 0);
  std::vector<int64_t> origins_storage(
      static_cast<size_t>(kInferOutputsBufferInt64s), 0);
  PluginCompilationOutputs outputs{};
  outputs.shapes_buf = shapes_storage.data();
  outputs.shapes_capacity = kInferOutputsBufferInt64s;
  outputs.shapes_needed = 0;
  outputs.origins_buf = origins_storage.data();
  outputs.origins_capacity = kInferOutputsBufferInt64s;
  outputs.origins_needed = 0;

  // Call the function with binary-safe parameters
  MY_LOG(2) << "About to call func with size = " << mlir_bytecode.size();

  CompilerError error = {};
  auto result =
      func(mlir_bytecode.data(), mlir_bytecode.size(), temp_output_path.c_str(),
           options_json.c_str(), &error, fs, &outputs);

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

  // Clean up temporary file (DLL is already in memory).
  std::remove(temp_output_path.c_str());

  // Build artifact.
  CompilationArtifact artifact;
  artifact.filename = "model_compiled";
  artifact.bytes = std::move(buffer);
  artifact.format = config.artifactFormat;

  // Parse the InferOnnxShapes output buffers the compiler just wrote.
  // The compiler always populates `*_needed` even when *_buf is too small;
  // we pre-allocated generously (see kInferOutputsBufferInt64s) so
  // truncation only happens if a future model pushes past the budget —
  // warn loudly in that case so the constant gets bumped (the data we
  // already have is still well-formed up to the truncation point, but
  // some output dims may be missing their origins).
  if (outputs.shapes_needed > outputs.shapes_capacity) {
    LOG(WARNING) << "MlirCompiler: refined_output_shapes truncated by "
                 << (outputs.shapes_needed - outputs.shapes_capacity)
                 << " int64s (needed=" << outputs.shapes_needed
                 << ", capacity=" << outputs.shapes_capacity
                 << "). Bump kInferOutputsBufferInt64s in MlirCompiler.cpp.";
  }
  if (outputs.origins_needed > outputs.origins_capacity) {
    LOG(WARNING) << "MlirCompiler: refined_output_dim_origins truncated by "
                 << (outputs.origins_needed - outputs.origins_capacity)
                 << " int64s (needed=" << outputs.origins_needed
                 << ", capacity=" << outputs.origins_capacity
                 << "). Bump kInferOutputsBufferInt64s in MlirCompiler.cpp.";
  }

  int64_t shapes_avail =
      std::min<int64_t>(outputs.shapes_needed, outputs.shapes_capacity);
  if (shapes_avail > 0) {
    int64_t cursor = 0;
    int64_t num_outputs = outputs.shapes_buf[cursor++];
    artifact.refined_output_shapes.reserve(static_cast<size_t>(num_outputs));
    for (int64_t i = 0; i < num_outputs && cursor < shapes_avail; ++i) {
      int64_t num_dims = outputs.shapes_buf[cursor++];
      std::vector<int64_t> dims;
      dims.reserve(static_cast<size_t>(num_dims));
      for (int64_t d = 0; d < num_dims && cursor < shapes_avail; ++d)
        dims.push_back(outputs.shapes_buf[cursor++]);
      artifact.refined_output_shapes.push_back(std::move(dims));
    }
    MY_LOG(2) << "Got " << artifact.refined_output_shapes.size()
              << " refined output shapes from compiler";
  }

  int64_t origins_avail =
      std::min<int64_t>(outputs.origins_needed, outputs.origins_capacity);
  if (origins_avail > 0) {
    // Triple stride: (arg, dim, mult_bits) per dim, plus one num_dims
    // marker per output, plus one num_outputs marker. `mult_bits` is the
    // IEEE 754 binary64 bit pattern of a double — bit-cast back here.
    int64_t cursor = 0;
    int64_t num_outputs = outputs.origins_buf[cursor++];
    artifact.refined_output_dim_origins.reserve(
        static_cast<size_t>(num_outputs));
    for (int64_t i = 0; i < num_outputs && cursor < origins_avail; ++i) {
      int64_t num_dims = outputs.origins_buf[cursor++];
      std::vector<CompilationArtifact::DimOriginTriple> dims;
      dims.reserve(static_cast<size_t>(num_dims));
      for (int64_t d = 0; d < num_dims && cursor + 2 < origins_avail; ++d) {
        int64_t arg = outputs.origins_buf[cursor++];
        int64_t idx = outputs.origins_buf[cursor++];
        int64_t mult_bits = outputs.origins_buf[cursor++];
        double mult;
        std::memcpy(&mult, &mult_bits, sizeof(double));
        dims.push_back({arg, idx, mult});
      }
      artifact.refined_output_dim_origins.push_back(std::move(dims));
    }
    MY_LOG(2) << "Got " << artifact.refined_output_dim_origins.size()
              << " refined output dim-origin maps from compiler";
  }

  LOG(INFO) << "Artifact created: " << artifact.bytes.size() << " bytes";

  return artifact;
}

} // namespace hipdnn::level1pass
