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

#include <morphizen-foundation/file_io.hpp>

namespace hipdnn::level1pass {

// Artifact format (native DLL or LLVM IR)
enum class ArtifactFormat { Native, LlvmIr };

// Compilation configuration
//
// `skipConstantData` selects the OnnxToHip finalize output:
//   * true  -> per-entry source (descriptors baked into __metadata_blob;
//              MlirCustomOp ctor streams constants into the GPU blob)
//   * false -> sidecar  (model.constants.bin + .json; MlirCustomOp ctor
//              goes through inference_init -> init_with_fs bulk hipMemcpy)
// The in-process EP path decides this inside pass_main::load_config based
// on ep.context_enable; EPContext export forces sidecar. The struct default
// (true / streaming) only applies to code paths that bypass load_config.
struct CompilationConfig {
  ArtifactFormat artifactFormat;
  int optLevel;
  bool skipConstantData = true;
};

// Compiled artifact (bytes + metadata)
struct CompilationArtifact {
  std::string filename;
  std::vector<uint8_t> bytes;
  ArtifactFormat format;
};

// Result of a compilation attempt. On success `artifact` is populated.
// On failure `error_message` carries the diagnostic from the hip-compiler
// plugin (or the higher-level reason — plugin missing, file IO failure, ...).
// Used by pass_main.cpp to surface compile failures rather than silently
// falling back to another EP. See the "Phase 0 — strict fallback" gotcha in
// CLAUDE.md for the rationale.
struct CompilationResult {
  std::optional<CompilationArtifact> artifact;
  std::string error_message;
};

/**
 * Simplified MLIR compiler that uses the hip-compiler plugin C API.
 *
 * Replaces the old direct LLVM/MLIR integration (MlirParser, MlirTransformer,
 * LlvmCompiler).
 *
 * NOTE: Mock runtime is not supported. The hip-compiler plugin always
 * generates native code that targets the actual HIP/ROCm runtime.
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
   * @return               CompilationResult — on success `.artifact` is set;
   *                       on failure `.error_message` carries the diagnostic
   *                       from the hip-compiler plugin (or a higher-level
   *                       failure description) so callers can surface it
   *                       loudly instead of silently dropping the subgraph.
   */
  static CompilationResult compileFromBytecode(const std::string &mlir_bytecode,
                                               const CompilationConfig &config,
                                               morphizen::FileSystem *fs);
};

} // namespace hipdnn::level1pass

#endif // HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
