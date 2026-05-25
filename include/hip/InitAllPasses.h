/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_INITALLPASSES_H
#define HIP_COMPILER_INITALLPASSES_H

#include "hip/Conversion/Passes.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"

namespace hip::compiler {

namespace detail {
/// Minimal ONNX dialect stub that claims the "onnx" namespace and permits
/// unknown operations.  This avoids depending on the full onnx-mlir library.
class OnnxStubDialect : public mlir::Dialect {
public:
  explicit OnnxStubDialect(mlir::MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                mlir::TypeID::get<OnnxStubDialect>()) {
    allowUnknownOperations();
  }
  static constexpr llvm::StringLiteral getDialectNamespace() { return "onnx"; }
};
} // namespace detail

/// Register all required dialects into a DialectRegistry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<detail::OnnxStubDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);

  // BufferDeallocationOpInterface external models. The
  // ownership-based buffer deallocation pass (run from
  // buildBufferDeallocationPipeline in Pipelines.cpp) queries this
  // interface on every op it visits and LLVM_FATALs out if the dialect
  // PROMISED the interface (via the canonical pass setup elsewhere in
  // MLIR) but never had its external-model implementation registered.
  // Arith is the canonical triggering case in our pipeline: when
  // OnnxToHip emits an `arith.constant <i64>` for a Range / Cast
  // operand, the bufferization wraps it in a buffer-producing chain
  // and the deallocation pass needs the interface to decide
  // ownership. Same for ControlFlow / SCF / GPU when those op-trees
  // appear (today they don't, but the registration is cheap and
  // matches mlir-opt's behavior, so we register them all up-front to
  // forestall the same crash class as new ops land).
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
}

/// Load all required dialects into an MLIRContext.
inline void loadAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

/// Register all passes and pipelines.
inline void registerAllPasses() {
  mlir::registerCanonicalizerPass();
  mlir::hip::registerOptimizeMemRefsPass();
  mlir::hip::registerPoolAllocsPass();
  mlir::hip::registerSimplifyOnnxPass();
  registerConversionPasses();
}

} // namespace hip::compiler

#endif // HIP_COMPILER_INITALLPASSES_H
