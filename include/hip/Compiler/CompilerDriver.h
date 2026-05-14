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

/// End-to-end compilation driver for MLIR → LLVM bitcode.
///
/// Orchestrates the complete compilation pipeline:
/// - MLIR pass pipeline (ONNX→HIP→LLVM→Interface)
/// - LLVM IR translation, runtime.bc linking, and optimization
/// - LLVM bitcode emission (consumed by BitcodeJIT in the EP DLL)
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

  bool emitBitcode(llvm::Module *llvmModule, const std::string &outputPath,
                   std::string &error_message);

  morphizen::FileSystem *fileSystem_ = nullptr;
  void *hipdnnHandle_ = nullptr;
  mlir::hip::CompiledGraphMap compiledGraphs_;
};

} // namespace hip::compiler

#endif // HIP_COMPILER_COMPILER_DRIVER_H
