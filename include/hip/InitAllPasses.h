/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_INITALLPASSES_H
#define UDNA_COMPILER_INITALLPASSES_H

#include "hip/Compiler/Passes/Passes.h"
#include "hip/Compiler/Pipeline.h"
#include "hip/Conversion/Passes.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

namespace udna::compiler {

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
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::ONNXDialect>();
  // Standard dialect bufferization models required by one-shot-bufferize.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  // Attach bufferization external models for all HIP compute ops so that
  // one-shot-bufferize can lower them without hip.alloc/hip.copy.
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);
}

/// Load all required dialects into an MLIRContext.
/// Use this for libraries like CompilerDriver that directly create MLIRContext.
/// Uses a DialectRegistry so that external bufferization models are attached
/// before the dialects are fully loaded.
inline void loadAllDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
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
  mlir::hip::registerOptimizeMemRefsPass();
  mlir::hip::registerPoolAllocsPass();
  registerConversionPasses();
}

} // namespace udna::compiler

#endif // UDNA_COMPILER_INITALLPASSES_H
