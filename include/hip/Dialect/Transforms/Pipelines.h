/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_TRANSFORMS_PIPELINES_H
#define HIP_DIALECT_TRANSFORMS_PIPELINES_H

#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

struct hipdnnHandle;
typedef hipdnnHandle *hipdnnHandle_t;

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

/// Default minimum number of tensor elements for constant externalization.
/// Set to 1 means all tensor constants are written to constants.bin
/// rather than inlined in the DLL, because inlining element tensors that
/// flow into a kernel as device operands turns a device buffer into a host
/// `arith.constant` — the GPU then dereferences a host pointer and faults
/// (historically surfaced as launch error 719; also reproduces as a GPU
/// "Memory access fault").
///
/// NOTE: a pure SIZE threshold cannot safely inline shape metadata. Raising
/// this to 16 (to keep tiny `Reshape`/`Range` shape scalars inline for
/// post-conversion shape inference) was tried and reintroduced the fault: a
/// sub-threshold attention scale scalar (a `mul` by a `[1,1,1,1]` operand) is
/// also a small constant, got inlined as host data, and the consuming kernel
/// page-faulted dereferencing the host pointer. Distinguishing a shape scalar
/// (safe to inline) from a data scalar (must stay externalized) requires a
/// USE-based decision, not a size one. Keep this at 1; any dynamic shape dim
/// that an externalized scalar would have carried is recovered post-conversion
/// by the dialect-level `--hip-infer-shapes` pass + canonicalize/cse, and the
/// residual runtime reads go through a synchronized D2H readback (`hip.readback_dim`,
/// using hipMemcpyDefault so it works whether the source scalar lives in device
/// or host-accessible memory).
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
  Option<bool> useOutputAllocator{
      *this, "use-output-allocator",
      llvm::cl::desc(
          "Allocator pipeline: replace buffer-results-to-out-params with "
          "hip-use-output-allocator so graph outputs are allocated in-graph "
          "via "
          "hip.alloc_output (and the hipdnn.use_output_allocator module "
          "attribute is "
          "set for the LLVM half to read) (default: false = classic "
          "out-params)"),
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
  // No use-output-allocator option here: convert-hip-to-llvm and
  // generate-interface read the `hipdnn.use_output_allocator` module attribute
  // set by hip-use-output-allocator in the ONNX-to-HIP half. When this pipeline
  // is invoked standalone in allocator mode, the input IR must already carry
  // that attribute.
};

/// Build the ONNX-to-HIP compilation pipeline.
///
/// Converts ONNX-level tensor IR into fully bufferized HIP memref IR with
/// pooled allocations and resolved extern constants.
///
/// \p fs -- when non-null, externalized constants are written through this
///   FileSystem (EPContext archive). When null, a DiskFileSystem is used.
void buildOnnxToHipPipeline(OpPassManager &pm,
                            const OnnxToHipPipelineOptions &options,
                            morphizen::FileSystem *fs = nullptr);

/// Build the ONNX-to-HIP pipeline with hipDNN graph compilation support.
///
/// Same as the FileSystem overload, but additionally inserts the
/// ConvertOnnxToHipDNN pass when handle is non-null. Supported ONNX ops
/// are compiled into hipDNN graphs at pass time; unsupported ops pass
/// through to ConvertOnnxToHip.
void buildOnnxToHipPipeline(OpPassManager &pm,
                            const OnnxToHipPipelineOptions &options,
                            morphizen::FileSystem *fs, hipdnnHandle_t handle,
                            CompiledGraphMap output_graphs);

/// Build the HIP-to-LLVM lowering pipeline. This is a separate pipeline
/// (not part of buildOnnxToHipPipeline) because the LLVM lowering is only
/// needed when producing executables via hip-compiler, not when inspecting
/// intermediate HIP memref IR via hip-mlir-opt.
///
/// The pipeline lowers HIP dialect ops to LLVM IR and appends a
/// GenerateInterface pass that creates four C-ABI wrapper functions
/// (inference_init, inference_compute, inference_cleanup,
/// inference_get_metadata_json).
void buildHipToLLVMPipeline(OpPassManager &pm,
                            const HipToLLVMPipelineOptions &options);

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
  Option<bool> useOutputAllocator{
      *this, "use-output-allocator",
      llvm::cl::desc(
          "Allocator pipeline: route the ONNX-to-HIP half through "
          "hip-use-output-allocator, which sets the "
          "hipdnn.use_output_allocator "
          "module attribute; convert-hip-to-llvm + generate-interface then "
          "read "
          "that attribute (default: false = classic out-params)"),
      llvm::cl::init(false)};
};

/// Build the complete HIPDNN pipeline: ONNX→HIP→LLVM→Interface.
/// Chains buildOnnxToHipPipeline and buildHipToLLVMPipeline.
void buildHipdnnPipeline(OpPassManager &pm,
                         const HipdnnPipelineOptions &options);

/// Register all pipelines with MLIR's global pass registry so they appear
/// in hip-mlir-opt --help and are usable as single-flag invocations.
/// Follows the torch-mlir PassPipelineRegistration pattern.
void registerHipPipelines();

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_TRANSFORMS_PIPELINES_H
