/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef COMPILATION_CONFIG_H
#define COMPILATION_CONFIG_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
class PassContext;
}

namespace hipdnn {
namespace level1pass {

/**
 * Output format for compiled artifacts.
 */
enum class ArtifactFormat { Native, LlvmIr };

/**
 * Configuration for MLIR compilation.
 */
struct CompilationConfig {
  ArtifactFormat artifactFormat;
  int optLevel;
  std::string outputFilename;

  static CompilationConfig
  fromProviderOptions(const std::shared_ptr<morphizen::PassContext> &context);
  static CompilationConfig defaultConfig();
};

/**
 * Result of MLIR compilation containing the compiled artifact and metadata.
 *
 * This struct holds:
 * - filename: Output filename (e.g., "model_compiled.dll")
 * - bytes: Raw binary content of the compiled artifact
 * - format: Native (DLL) or LLVM IR
 * - compilation_ms: Time spent in MLIR→LLVM IR compilation
 * - linking_ms: Time spent in LLVM IR→DLL linking
 */
struct CompilationArtifact {
  std::string filename;         // e.g., "model_compiled.dll"
  std::vector<uint8_t> bytes;   // Raw binary content
  ArtifactFormat format;        // Native or LlvmIr
  int64_t compilation_ms;       // MLIR→LLVM IR compilation time
  int64_t linking_ms;           // LLVM IR→DLL linking time
};

} // namespace level1pass
} // namespace hipdnn

#endif
