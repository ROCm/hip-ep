/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef MORPHIZEN_MLIR_COMPILER_COMPILER_DRIVER_H
#define MORPHIZEN_MLIR_COMPILER_COMPILER_DRIVER_H

#include "compilation_options.pb.h"
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

namespace morphizen::mlir_compiler {

/**
 * End-to-end compilation driver for MLIR → DLL/Object/IR.
 *
 * Orchestrates the complete compilation pipeline:
 * - MLIR pass pipeline (ONNX→HIP→LLVM→Interface)
 * - LLVM IR translation and optimization
 * - Object file generation
 * - DLL linking
 *
 * Used by:
 * - morphizen-compile (CLI tool)
 * - CInterface (DLL C API)
 * - Morphizen execution provider plugin
 */
class CompilerDriver {
public:
  CompilerDriver() = default;
  ~CompilerDriver() = default;

  /**
   * Compile MLIR string to output file.
   *
   * @param input_mlir MLIR text/bytecode input (zero-copy, binary-safe)
   * @param output_path Output file path (.dll/.obj/.ll)
   * @param options Compilation options (proto message)
   * @param error_message Output parameter for error details
   * @return true on success, false on error (check error_message)
   */
  bool compile(llvm::StringRef input_mlir, const std::string& output_path,
               const morphizen::mlir_compiler::CompilationOptions& options,
               std::string& error_message);

  /**
   * Compile MLIR module to output file.
   *
   * @param module Parsed MLIR module
   * @param output_path Output file path (.dll/.obj/.ll)
   * @param options Compilation options (proto message)
   * @param error_message Output parameter for error details
   * @return true on success, false on error (check error_message)
   */
  bool
  compileFromModule(mlir::ModuleOp module, const std::string& output_path,
                    const morphizen::mlir_compiler::CompilationOptions& options,
                    std::string& error_message);

  /**
   * Validate MLIR input without compiling.
   *
   * Fast check - parses and validates but doesn't run full pipeline.
   *
   * @param input_mlir MLIR text/bytecode input (zero-copy, binary-safe)
   * @param error_message Output parameter for error details
   * @return true if valid, false if invalid (check error_message)
   */
  bool validate(llvm::StringRef input_mlir, std::string& error_message);

private:
  /**
   * Core compilation implementation (shared by compile and compileFromModule).
   */
  bool compileImpl(mlir::ModuleOp module, const std::string& output_path,
                   const morphizen::mlir_compiler::CompilationOptions& options,
                   std::string& error_message);

  /**
   * Run MLIR transformation passes.
   */
  bool
  runMLIRPasses(mlir::ModuleOp module,
                const morphizen::mlir_compiler::CompilationOptions& options,
                std::string& error_message);

  /**
   * Translate MLIR to LLVM IR.
   */
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& llvmContext,
                    std::string& error_message);

  /**
   * Link runtime module for zero-cost abstraction.
   */
  bool linkRuntime(llvm::Module* llvmModule, std::string& error_message);

  /**
   * Optimize LLVM IR.
   */
  void optimizeLLVMIR(llvm::Module* llvmModule, int optLevel);

  /**
   * Emit LLVM IR to file.
   */
  bool emitLLVMIR(llvm::Module* llvmModule, const std::string& outputPath,
                  std::string& error_message);

  /**
   * Compile LLVM IR to object file.
   */
  bool compileToObject(llvm::Module* llvmModule, const std::string& outputPath,
                       std::string& error_message);

  /**
   * Link object file to DLL.
   */
  bool linkToDLL(const std::string& objPath, const std::string& dllPath,
                 const std::vector<std::string>& libraries,
                 const std::vector<std::string>& library_paths,
                 const std::vector<std::string>& export_symbols,
                 std::string& error_message);

  /**
   * Clean up intermediate files.
   */
  void cleanupIntermediates(const std::string& basePath);
};

} // namespace morphizen::mlir_compiler

#endif // MORPHIZEN_MLIR_COMPILER_COMPILER_DRIVER_H
