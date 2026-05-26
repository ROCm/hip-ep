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
  // Per-output (in declaration order), per-dim refined shape from the
  // MLIR-level `InferOnnxShapes` pre-lowering pass. Positive integers
  // are static dims; -1 is genuinely dynamic. Empty when the compiler
  // didn't emit refined shapes (older plugin or empty graph). Consumed
  // by `pass_main.cpp::build_metadata_json` to populate
  // `DimSource.static_value`.
  std::vector<std::vector<int64_t>> refined_output_shapes;
  // Per-output (in declaration order), per-dim SSA origin. Each triple
  // is `(graph_arg_index, dim_idx, mult)` — runtime computes
  // `round(inputs[arg].shape[dim] * mult)`. `mult == 1.0` is the
  // identity passthrough (most LLM dynshape outputs); `mult == 1/K`
  // covers Reshape-induced spatial mergers (e.g. Qwen vision's
  // `num_patches -> num_patches/4` patch merger contributes mult=0.25).
  // `(-1, -1, 1.0)` means no traceable origin. Populated by
  // InferOnnxShapes' backward-trace from each function output. Consumed
  // by `pass_main.cpp::build_metadata_json` to populate
  // `DimSource.input_idx + dim_idx + mult` for dynamic output dims whose
  // dim_param names don't match any input dim_param — which lets the
  // EP work on models that ship outputs with auto-generated /
  // semantically-different symbolic names (e.g. Gemma-3 vision.onnx's
  // `image_features: [num_image_tokens, MatMulimage_features_dim_1,
  // 2560]` vs the input's `pixel_values: [num_images, 3, 896, 896]`)
  // WITHOUT modifying the ONNX file.
  struct DimOriginTriple {
    int64_t arg_idx;
    int64_t dim_idx;
    double mult;
  };
  std::vector<std::vector<DimOriginTriple>> refined_output_dim_origins;
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
