/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
#define HIPDNN_LEVEL1PASS_MLIR_COMPILER_H

#include "CompilationArtifact.h"
#include "CompilationConfig.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn {
namespace level1pass {

/**
 * Simplified MLIR compiler that uses morphizen-mlir-compiler plugin C API.
 *
 * Replaces the old direct LLVM/MLIR integration (MlirParser, MlirTransformer,
 * LlvmCompiler).
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

} // namespace level1pass
} // namespace hipdnn

#endif // HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
