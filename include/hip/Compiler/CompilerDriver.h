/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_COMPILER_DRIVER_H
#define HIP_COMPILER_COMPILER_DRIVER_H

#include "compilation_options_generated.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace hip::compiler {

/// One traced origin for a dynamic output dim. Encodes "this dim's runtime
/// value is `round(inputs[arg_idx].shape[dim_idx] * mult)`". Mirrors the
/// `DimOriginInfo` produced by `lib/Conversion/OnnxToHip/InferOnnxShapes.cpp`;
/// duplicated here to avoid leaking MLIR types into the public driver header.
/// `arg_idx == -1` is the "no traceable origin" sentinel.
struct DimOriginTriple {
  int64_t arg_idx;
  int64_t dim_idx;
  double mult;
};

/// End-to-end compilation driver for MLIR → DLL/Object/IR.
///
/// Orchestrates the complete compilation pipeline:
/// - MLIR pass pipeline (ONNX→HIP→LLVM→Interface)
/// - LLVM IR translation and optimization
/// - Object file generation
/// - DLL linking
class CompilerDriver {
public:
  CompilerDriver() = default;
  ~CompilerDriver() = default;

  /// Set FileSystem for external constant storage.
  /// When set, onnx.Constant data is written to "constants.bin" via fs
  /// instead of being embedded in the DLL. Must be called before compile().
  void setFileSystem(morphizen::FileSystem *fs) { fileSystem_ = fs; }

  /// Set hipDNN handle for graph compilation support.
  /// When set, the ConvertOnnxToHipDNN pass is inserted into the pipeline
  /// to compile supported ONNX ops into hipDNN graphs. Must be called
  /// before compile().
  void setHipdnnHandle(void *handle) { hipdnnHandle_ = handle; }

  /// Retrieve compiled hipDNN graphs after compile() returns.
  /// The returned map contains graphs compiled by the ConvertOnnxToHipDNN
  /// pass. Caller should store them in a registry for runtime execution.
  mlir::hip::CompiledGraphMap getCompiledGraphs() { return compiledGraphs_; }

  /// Compile MLIR string to output file.
  bool compile(llvm::StringRef input_mlir, const std::string &output_path,
               const mlir::hip::CompilationOptionsT &options,
               std::string &error_message);

  /// Compile MLIR module to output file.
  bool compileFromModule(mlir::ModuleOp module, const std::string &output_path,
                         const mlir::hip::CompilationOptionsT &options,
                         std::string &error_message);

  /// Validate MLIR input without compiling.
  bool validate(llvm::StringRef input_mlir, std::string &error_message);

  /// After a successful `compile()`, returns per-function-result refined
  /// shapes (outer vec by result index, inner by dim — positive ints for
  /// static dims, -1 for genuinely dynamic). Populated by reading the
  /// module attributes that `InferOnnxShapes` attaches inside the pass
  /// pipeline; captures the snapshot WHILE the function is still
  /// `func::FuncOp` (before HipToLLVM converts it to `llvm.func`).
  const std::vector<std::vector<int64_t>> &refinedOutputShapes() const {
    return refinedOutputShapes_;
  }

  /// After a successful `compile()`, returns per-result, per-dim SSA
  /// origins traced from `InferOnnxShapes`. Same indexing convention as
  /// `refinedOutputShapes()`.
  const std::vector<std::vector<DimOriginTriple>> &
  refinedOutputDimOrigins() const {
    return refinedOutputDimOrigins_;
  }

private:
  bool compileImpl(mlir::ModuleOp module, const std::string &output_path,
                   const mlir::hip::CompilationOptionsT &options,
                   std::string &error_message);

  bool runMLIRPasses(mlir::ModuleOp module,
                     const mlir::hip::CompilationOptionsT &options,
                     std::string &error_message);

  std::unique_ptr<llvm::Module>
  translateToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext &llvmContext,
                    std::string &error_message);

  bool linkRuntime(llvm::Module *llvmModule, std::string &error_message);

  void optimizeLLVMIR(llvm::Module *llvmModule, int optLevel);

  bool emitLLVMIR(llvm::Module *llvmModule, const std::string &outputPath,
                  std::string &error_message);

  bool compileToObject(llvm::Module *llvmModule, const std::string &outputPath,
                       std::string &error_message);

  bool linkToDLL(const std::string &objPath, const std::string &dllPath,
                 const std::vector<std::string> &libraries,
                 const std::vector<std::string> &library_paths,
                 const std::vector<std::string> &export_symbols,
                 std::string &error_message);

  /// Discover GPU runtime libraries from THEROCK_DIST environment variable.
  void discoverLibraries(std::vector<std::string> &libraries,
                         std::vector<std::string> &library_paths);

  void cleanupIntermediates(const std::string &basePath);

  /// Read module-level attributes attached by `InferOnnxShapes` and
  /// populate `refinedOutputShapes_` / `refinedOutputDimOrigins_`. Called
  /// inside `compileImpl` once `runMLIRPasses` succeeds — the module is
  /// still in memory at that point and the attributes survive the
  /// `func.func → llvm.func` conversion. No-op when the attributes are
  /// missing (e.g. a future pipeline that skips InferOnnxShapes).
  void readRefinedOutputsFromModule(mlir::ModuleOp module);

  morphizen::FileSystem *fileSystem_ = nullptr;
  void *hipdnnHandle_ = nullptr;
  mlir::hip::CompiledGraphMap compiledGraphs_;
  std::vector<std::vector<int64_t>> refinedOutputShapes_;
  std::vector<std::vector<DimOriginTriple>> refinedOutputDimOrigins_;
};

} // namespace hip::compiler

#endif // HIP_COMPILER_COMPILER_DRIVER_H
