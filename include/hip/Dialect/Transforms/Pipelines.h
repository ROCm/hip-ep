/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PIPELINES_H
#define HIP_PIPELINES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hip {

/// Pipeline options forwarded to the ConvertOnnxToHipPass for constant
/// externalization. These mirror the pass options on ConvertOnnxToHipPass
/// so the pipeline flag surface is:
///   --hip-onnx-to-hip-pipeline='externalize-min-num-elements=256
///                                externalize-output-dir=/tmp'
struct OnnxToHipPipelineOptions
    : public PassPipelineOptions<OnnxToHipPipelineOptions> {
  Option<std::string> externalizeOutputDir{
      *this, "externalize-output-dir",
      llvm::cl::desc(
          "Directory for sidecar .constants.bin/.json files (empty = cwd)"),
      llvm::cl::init("")};
  Option<int64_t> externalizeMinNumElements{
      *this, "externalize-min-num-elements",
      llvm::cl::desc(
          "Minimum number of tensor elements to externalize (0 = disabled)"),
      llvm::cl::init(0)};
};

/// Build the ONNX-to-HIP compilation pipeline. This is the main pipeline
/// that takes ONNX-level tensor IR and produces fully bufferized HIP memref
/// IR with pooled allocations and resolved extern constants. The pipeline
/// order is critical -- see Pipelines.cpp for detailed commentary on why
/// each pass must appear at its position.
void buildOnnxToHipPipeline(OpPassManager &pm,
                            const OnnxToHipPipelineOptions &options);

/// Build the HIP-to-LLVM lowering pipeline. This is a separate pipeline
/// (not part of buildOnnxToHipPipeline) because the LLVM lowering is only
/// needed when producing executables via hip-compiler, not when inspecting
/// intermediate HIP memref IR via hip-mlir-opt.
void buildHipToLLVMPipeline(OpPassManager &pm);

/// Register both pipelines with MLIR's global pass registry so they appear
/// in hip-mlir-opt --help and are usable as single-flag invocations.
/// Follows the torch-mlir PassPipelineRegistration pattern.
void registerHipPipelines();

} // namespace hip
} // namespace mlir

#endif // HIP_PIPELINES_H
