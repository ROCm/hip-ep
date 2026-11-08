/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_TRANSFORMS_PIPELINES_H
#define HIP_DIALECT_TRANSFORMS_PIPELINES_H

#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"
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
/// residual runtime reads go through a synchronized D2H readback
/// (`hip.readback_dim`, using hipMemcpyDefault so it works whether the source
/// scalar lives in device or host-accessible memory).
constexpr int64_t kDefaultExternalizeMinNumElements = 1;

/// Pipeline options forwarded to ExternalizeConstantsPass. ConvertOnnxToHipPass
/// only emits hip.constant carriers. These mirror the externalizer options
/// so the pipeline flag surface is:
///   --onnx-to-hip-pipeline='externalize-min-num-elements=256
///                                externalize-output-dir=/tmp'
struct OnnxToHipPipelineOptions
    : public PassPipelineOptions<OnnxToHipPipelineOptions> {
  Option<std::string> externalizeOutputDir{
      *this, "externalize-output-dir",
      llvm::cl::desc("Directory for constants-file .constants.bin/.json files "
                     "(empty = cwd)"),
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

/// Build the common tail of the ONNX-to-HIP pipeline: everything after the
/// OnnxToHip conversion (shape inference, HIP-to-HIP pattern rewriting,
/// constant externalization, bufferization, output-allocator rewrite, pooling,
/// extern-constant resolution) up to -- but not including -- the HIP-to-LLVM
/// lowering. Exposed
/// so tools that build a custom head (e.g. hip-rocmlir-compiler, which inserts
/// the rocMLIR fusion + a rocMLIR compile/embed step) can run the standard tail
/// without duplicating its load-bearing pass ordering.
void buildOnnxToHipPipelineTail(OpPassManager &pm,
                                const OnnxToHipPipelineOptions &options,
                                morphizen::FileSystem *fs = nullptr);

/// Build the head of the ONNX-to-HIP pipeline: everything up to and including
/// the OnnxToHip conversion. Split out from the tail so a caller can insert
/// work between the two -- the rocMLIR path outlines kernels and compiles them
/// there, which has to happen on tensor-level HIP IR before the tail
/// bufferizes. \p handle, when non-null, inserts the hipDNN graph passes.
void buildOnnxToHipPipelineHead(OpPassManager &pm,
                                hipdnnHandle_t handle = nullptr,
                                CompiledGraphMap output_graphs = {});

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
};

/// Build the complete HIPDNN pipeline: ONNX→HIP→LLVM→Interface.
/// Chains buildOnnxToHipPipeline and buildHipToLLVMPipeline.
void buildHipdnnPipeline(OpPassManager &pm,
                         const HipdnnPipelineOptions &options);

/// Add the rocMLIR kernel fusion to a module pass manager: the quantize
/// preparation, the grouping of the pointwise ops of main_graph into
/// hip.rocmlir_elementwise modules, the outline of each anchor with its
/// modules into a rock.kernel function, then the merge of duplicate kernels.
/// A non-empty reportPath
/// appends the --hip-rocmlir-fusion-report records to that file. Every caller
/// builds the fusion through this function so that the pass order and the
/// options stay the same across tools.
void addRocMlirFusionPasses(OpPassManager &pm,
                            const FuseRocMlirAnchorPassOptions &fuseOptions,
                            llvm::StringRef reportPath = "");

struct RocMlirFusionPipelineOptions
    : PassPipelineOptions<RocMlirFusionPipelineOptions> {
  Option<std::string> arch{
      *this, "arch",
      llvm::cl::desc("Target GPU architecture stamped as rock.arch"),
      llvm::cl::init("")};
  ListOption<std::string> gemmGemmAccelTypes{
      *this, "gemm-gemm-accel-types",
      llvm::cl::desc("Gemm element types with matrix acceleration")};
  Option<std::string> report{
      *this, "report",
      llvm::cl::desc("Append the fusion report records to this file"),
      llvm::cl::init("")};
};

/// The rocMLIR fusion as one pipeline, for LIT and hip-mlir-opt.
void buildRocMlirFusionPipeline(OpPassManager &pm,
                                const RocMlirFusionPipelineOptions &options);

struct RocMlirPipelineOptions : PassPipelineOptions<RocMlirPipelineOptions> {};

/// Build rocmlirTriton pipeline
void buildRocMlirPipeline(OpPassManager &pm,
                          const RocMlirPipelineOptions &options);

/// Register all pipelines with MLIR's global pass registry so they appear
/// in hip-mlir-opt --help and are usable as single-flag invocations.
/// Follows the torch-mlir PassPipelineRegistration pattern.
void registerHipPipelines();

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_TRANSFORMS_PIPELINES_H
