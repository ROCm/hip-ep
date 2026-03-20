/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_COMPILER_DRIVER_H
#define HIP_COMPILER_COMPILER_DRIVER_H

#include "compilation_options_generated.h"
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

  morphizen::FileSystem *fileSystem_ = nullptr;
};

} // namespace hip::compiler

#endif // HIP_COMPILER_COMPILER_DRIVER_H
