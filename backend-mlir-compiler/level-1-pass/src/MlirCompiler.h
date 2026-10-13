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

// Per-model artifact format, selected by the `artifact_format` provider
// option (single compile option). Both are loaded by the EP via
// InferenceState::create:
//   * LlvmIr -> OS-portable LLVM IR (serialized as .bc), JIT-loaded in-process
//               by LlvmIrJit.
//   * Native -> per-OS native .dll/.so loaded via morphizen::Plugin
//               (LoadLibrary/dlopen). Opt-in; see
//               docs/native-vs-ir-comparison.md.
enum class ArtifactFormat { LLVM_IR, NATIVE };

// Compilation configuration
//
// `skipConstantData` selects the OnnxToHip finalize output:
//   * true  -> per-entry source (descriptors baked into __metadata_blob;
//              MlirCustomOp ctor streams constants into the GPU blob)
//   * false -> constants file  (model.constants.bin + .json; MlirCustomOp ctor
//              goes through inference_init -> init_with_fs bulk hipMemcpy)
// The in-process EP path decides this inside pass_main::load_config based
// on ep.context_enable; EPContext export forces constants file. The struct
// default (true / streaming) only applies to code paths that bypass
// load_config.
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

/**
 * MLIR compiler driver that dispatches to the hip-compiler plugin C API.
 *
 * `compileFromBytecode` serializes the provided MLIR bytecode, calls the
 * plugin's `hip_compile_with_fs`, and reads back the resulting per-model
 * LLVM bitcode artifact for inclusion in the EPContext tar. The downstream
 * EP loads it via LlvmIrJit.
 *
 * NOTE: Mock runtime is not supported. The hip-compiler plugin always
 * targets the real HIP/ROCm runtime; ML inference on a host without
 * ROCm is fundamentally out of scope for this driver.
 */
class MlirCompiler {
public:
  /**
   * Compile MLIR bytecode to a per-model LLVM bitcode artifact.
   *
   * @param mlir_bytecode  MLIR bytecode (as from Graph.save_string())
   * @param config         Compilation configuration
   * @param fs             FileSystem for externalized constants
   * @return               Compiled bitcode artifact (bytes + metadata),
   *                       or nullopt on failure
   */
  static std::optional<CompilationArtifact>
  compileFromBytecode(const std::string &mlir_bytecode,
                      const CompilationConfig &config,
                      morphizen::FileSystem *fs);
};

} // namespace hipdnn::level1pass

#endif // HIPDNN_LEVEL1PASS_MLIR_COMPILER_H
