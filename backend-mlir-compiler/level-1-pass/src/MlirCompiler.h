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
  // Output-allocator mode (2-arg inference_compute + in-graph hip.alloc_output)
  // is the only ABI at the EP front-end -- there is no provider option and no
  // classic out-param fallback. load_config sets this true unconditionally and
  // writes the SAME value into the model metadata so the EP's dispatch arity
  // always agrees with the compiled DLL's ABI. Default true so any path that
  // bypasses load_config still gets the supported ABI.
  bool useOutputAllocator = true;
};

// Compiled artifact (bytes + metadata)
struct CompilationArtifact {
  std::string filename;
  std::vector<uint8_t> bytes;
  ArtifactFormat format;
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
   * @return               Compiled artifact (bytes + metadata), or nullopt on
   * failure
   */
  static std::optional<CompilationArtifact>
  compileFromBytecode(const std::string &mlir_bytecode,
                      const CompilationConfig &config,
                      morphizen::FileSystem *fs);
};

} // namespace hipdnn::level1pass

#endif // HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
