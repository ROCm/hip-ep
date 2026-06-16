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

/// End-to-end compilation driver for MLIR → per-model artifact.
///
/// Orchestrates the complete compilation pipeline:
/// - MLIR pass pipeline (ONNX->HIP->LLVM->Interface)
/// - LLVM IR translation and target-independent optimization
/// - Artifact emission selected by CompilationOptions.output_mode:
///     * Bitcode (default): OS-portable LLVM bitcode (consumed by LlvmIrJit
///       in the EP DLL, which JIT-loads its own per-OS runtime.bc separately).
///     * Native: merge runtime.bc, emit a host object, and link a per-OS
///       .dll/.so via DLLLinker. Opt-in for benchmarking/dev.
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

  void optimizeLLVMIR(llvm::Module *llvmModule, int optLevel);

  // ---- Bitcode backend (default) ----
  bool emitLlvmIr(llvm::Module *llvmModule, const std::string &outputPath,
                  std::string &error_message);

  // ---- Native backend (output_mode == NATIVE) ----
  // Merge the embedded runtime.bc into the module before object codegen
  // (producer-time link; the bitcode path skips this for OS-portability).
  bool linkRuntime(llvm::Module *llvmModule, std::string &error_message);

  // Emit a host object file (.obj/.o).
  bool compileToObject(llvm::Module *llvmModule, const std::string &outputPath,
                       std::string &error_message);

  // Link the object into a per-OS .dll/.so exporting the 5 inference symbols.
  bool linkToDLL(const std::string &objPath, const std::string &dllPath,
                 const std::vector<std::string> &libraries,
                 const std::vector<std::string> &library_paths,
                 const std::vector<std::string> &export_symbols,
                 std::string &error_message);

  // Discover GPU runtime libraries (THEROCK_DIST), the per-arch custom-kernels
  // import lib, and the hipDNN graph runtime, for the native link step.
  void discoverLibraries(std::vector<std::string> &libraries,
                         std::vector<std::string> &library_paths);

  // Remove intermediate .ll/.obj files left by the native path.
  void cleanupIntermediates(const std::string &basePath);

  morphizen::FileSystem *fileSystem_ = nullptr;
  void *hipdnnHandle_ = nullptr;
  mlir::hip::CompiledGraphMap compiledGraphs_;
};

} // namespace hip::compiler

#endif // HIP_COMPILER_COMPILER_DRIVER_H
