/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H
#define MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H

#include "morphizen-mlir-compiler/Compiler/Passes/Passes.h"
#include "morphizen-mlir-compiler/Compiler/Pipeline.h"
#include "morphizen-mlir-compiler/Conversion/Passes.h"
#include "morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h"
#include "morphizen-mlir-compiler/Dialect/Hip/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

namespace morphizen {

// clang-format off
// Usage Guide:
// ============
//
// Component           | Uses registerAllDialects? | Uses loadAllDialects? | Why?
// --------------------|---------------------------|----------------------|----------------------------------
// morphizen-opt       | ✅ Yes                    | ❌ No                 | Generic tool using MlirOptMain
// morphizen-compile   | ❌ No                     | ❌ No (indirect)      | Delegates to CompilerDriver
// dll                 | ❌ No                     | ❌ No                 | Just DLL entry point
// CompilerDriver      | ❌ No                     | ✅ Yes                | Library doing actual compilation
//
// Key Distinction:
// - registerAllDialects(): For tools using MLIR's MlirOptMain (lazy loading via DialectRegistry)
// - loadAllDialects(): For libraries directly managing MLIRContext (eager loading)
// clang-format on

/// Register all required dialects into a DialectRegistry.
/// Use this for tools like morphizen-opt that use DialectRegistry.
inline void registerAllDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::ONNXDialect>();
}

/// Load all required dialects into an MLIRContext.
/// Use this for libraries like CompilerDriver that directly create MLIRContext.
inline void loadAllDialects(mlir::MLIRContext& context) {
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<mlir::hip::HipDialect>();
  context.loadDialect<mlir::ONNXDialect>();
}

/// Register all passes and pipelines for use with morphizen-opt.
/// This is a convenience function that registers:
/// - MLIR standard passes (canonicalizer)
/// - HIP transform passes (memory pooling, buffer deallocation)
/// - Conversion passes (ONNX→HIP, HIP→LLVM)
/// - Compiler passes (GenerateInterface)
/// - Morphizen pipeline (complete ONNX→HIP→LLVM→Interface)
inline void registerAllPasses() {
  // MLIR standard passes
  mlir::registerCanonicalizerPass();

  // Morphizen custom passes
  mlir::hip::registerHipTransformPasses();
  registerConversionPasses();
  compiler::registerCompilerPasses();
  compiler::registerMorphizenPipeline();
}

} // namespace morphizen

#endif // MORPHIZEN_MLIR_COMPILER_INITALLPASSES_H
