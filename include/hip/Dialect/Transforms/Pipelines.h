//===- Pipelines.h - HIP pass pipeline declarations ----------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
#ifndef HIP_DIALECT_TRANSFORMS_PIPELINES_H
#define HIP_DIALECT_TRANSFORMS_PIPELINES_H

#include "hip/Conversion/OnnxToHipDNN/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

struct hipdnnHandle;
typedef hipdnnHandle* hipdnnHandle_t;

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

/// Default minimum number of tensor elements for constant externalization.
/// Set to 1 means all tensor constants are written to constants.bin
/// rather than inlined in the DLL, because the inline element tensors
/// as scalar kernel arguments causes GPU launch failures (error 719)
constexpr int64_t kDefaultExternalizeMinNumElements = 1;

/// Pipeline options forwarded to the ConvertOnnxToHipPass for constant
/// externalization. These mirror the pass options on ConvertOnnxToHipPass
/// so the pipeline flag surface is:
///   --onnx-to-hip-pipeline='externalize-min-num-elements=256
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
  Option<bool> skipConstantData{
      *this, "skip-constant-data",
      llvm::cl::desc("Skip writing constant data to constants.bin (metadata "
                     "only). Used for ORT EP live-compile path."),
      llvm::cl::init(false)};
};

/// Pipeline options for the HIP-to-LLVM lowering pipeline.
/// Controls the GenerateInterface pass at the end of the pipeline.
struct HipToLLVMPipelineOptions
    : public PassPipelineOptions<HipToLLVMPipelineOptions> {
  Option<std::string> constantsFile{
      *this, "constants-file",
      llvm::cl::desc(
          "Constants filename embedded in metadata (default: constants.bin)"),
      llvm::cl::init("constants.bin")};
};

/// Build the ONNX-to-HIP compilation pipeline.
///
/// Converts ONNX-level tensor IR into fully bufferized HIP memref IR with
/// pooled allocations and resolved extern constants.
///
/// \p fs            When non-null, externalized constants are written through
///                  this FileSystem (e.g. an EPContext archive).  When null,
///                  a DiskFileSystem rooted at `options.externalizeOutputDir`
///                  is used.
/// \p handle        Optional live `hipdnnHandle_t`.  When non-null, the
///                  pipeline prepends the OutlineOnnxToHipDNN +
///                  CompileHipDNNGraphs passes so that supported ONNX ops are
///                  compiled into hipDNN graphs ahead of `--convert-onnx-to-
///                  hip`.  When null, those passes are skipped and the entire
///                  graph flows through the standard ONNX-to-HIP path.
/// \p output_graphs Out-parameter for the compiled graph table; required when
///                  \p handle is non-null and ignored otherwise.
void buildOnnxToHipPipeline(OpPassManager& pm,
                            const OnnxToHipPipelineOptions& options,
                            morphizen::FileSystem* fs = nullptr,
                            hipdnnHandle_t handle = nullptr,
                            CompiledGraphMap output_graphs = {});

/// Build the HIP-to-LLVM lowering pipeline. This is a separate pipeline
/// (not part of buildOnnxToHipPipeline) because the LLVM lowering is only
/// needed when producing executables via hip-compiler, not when inspecting
/// intermediate HIP memref IR via hip-mlir-opt.
///
/// The pipeline lowers HIP dialect ops to LLVM IR and appends a
/// GenerateInterface pass that creates four C-ABI wrapper functions
/// (inference_init, inference_compute, inference_cleanup,
/// inference_get_metadata_json).
void buildHipToLLVMPipeline(OpPassManager& pm,
                            const HipToLLVMPipelineOptions& options);

/// Combined pipeline options for the full ONNX→HIP→LLVM→Interface flow.
/// Used by hip-mlir-opt --hipdnn-pipeline and the compiler driver.
struct HipdnnPipelineOptions
    : public PassPipelineOptions<HipdnnPipelineOptions> {
  Option<std::string> constantsFile{
      *this, "constants-file",
      llvm::cl::desc("Filename for constants data embedded in module metadata "
                     "(default: constants.bin)"),
      llvm::cl::init("constants.bin")};
  Option<std::string> constantsDir{
      *this, "constants-dir",
      llvm::cl::desc("Directory to write constants file into (default: cwd)"),
      llvm::cl::init("")};
  Option<int64_t> externalizeMinNumElements{
      *this, "externalize-min-num-elements",
      llvm::cl::desc(
          "Minimum number of tensor elements to externalize (0 = disabled)"),
      llvm::cl::init(0)};
};

/// Build the complete HIPDNN pipeline: ONNX→HIP→LLVM→Interface.
/// Chains buildOnnxToHipPipeline and buildHipToLLVMPipeline.
void buildHipdnnPipeline(OpPassManager& pm,
                         const HipdnnPipelineOptions& options);

/// Register all pipelines with MLIR's global pass registry so they appear
/// in hip-mlir-opt --help and are usable as single-flag invocations.
/// Follows the torch-mlir PassPipelineRegistration pattern.
void registerHipPipelines();

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_TRANSFORMS_PIPELINES_H
