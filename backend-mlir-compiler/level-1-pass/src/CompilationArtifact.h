/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_LEVEL1PASS_COMPILATION_ARTIFACT_H
#define HIPDNN_LEVEL1PASS_COMPILATION_ARTIFACT_H

#include "CompilationConfig.h"
#include <cstdint>
#include <string>
#include <vector>

namespace hipdnn {
namespace level1pass {

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
  std::string filename;                     // e.g., "model_compiled.dll"
  std::vector<uint8_t> bytes;               // Raw binary content
  CompilationConfig::ArtifactFormat format; // Native or LlvmIr
  int64_t compilation_ms;                   // MLIR→LLVM IR compilation time
  int64_t linking_ms;                       // LLVM IR→DLL linking time
};

} // namespace level1pass
} // namespace hipdnn

#endif // HIPDNN_LEVEL1PASS_COMPILATION_ARTIFACT_H
