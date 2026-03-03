/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
#define HIPDNN_LEVEL1PASS_MLIR_COMPILER_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn::level1pass {

// Artifact format (native DLL or LLVM IR)
enum class ArtifactFormat { Native, LlvmIr };

// Compilation configuration
struct CompilationConfig {
  ArtifactFormat artifactFormat;
  int optLevel;
};

// Compiled artifact (bytes + metadata)
struct CompilationArtifact {
  std::string filename;
  std::vector<uint8_t> bytes;
  ArtifactFormat format;
};

/**
 * Simplified MLIR compiler that uses UDNA-compiler plugin C API.
 *
 * Replaces the old direct LLVM/MLIR integration (MlirParser, MlirTransformer,
 * LlvmCompiler).
 *
 * NOTE: Mock runtime is not supported. The UDNA-compiler plugin
 * always generates native code that targets the actual HIP/ROCm runtime.
 * Mock runtime functionality was removed as it is not compatible with the
 * plugin-based compilation architecture.
 */
class MlirCompiler {
public:
  /**
   * Compile MLIR bytecode to native artifact (DLL).
   *
   * @param mlir_bytecode  MLIR bytecode (as from Graph.save_string())
   * @param config         Compilation configuration
   * @return               Compiled artifact (bytes + metadata), or nullopt on
   * failure
   */
  static std::optional<CompilationArtifact>
  compileFromBytecode(const std::string &mlir_bytecode,
                      const CompilationConfig &config);
};

} // namespace hipdnn::level1pass

#endif // HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
