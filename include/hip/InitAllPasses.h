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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
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
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<detail::OnnxStubDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  // Register BufferDeallocationOpInterface external model for arith — required
  // by the upstream buildBufferDeallocationPipeline when arith.constant /
  // arith.select / arith.constant_index appear in bufferized IR (e.g. via
  // rank-1 size-1 scalar extraction in Range lowering).
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // linalg.* ops (in particular linalg.map / linalg.fill emitted by the
  // upstream tensor bufferization of tensor.splat / tensor.empty + Where)
  // need their own external bufferizable interface model.  Without this
  // the OneShotBufferizePass fails on graphs containing onnx.ConstantOfShape
  // -> tensor.splat -> linalg.map (e.g. multimodal embedding.onnx).
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);
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
