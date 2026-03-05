/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef hip_COMPILER_DIALECT_HIP_IR_HIPBUFFERIZE_H
#define hip_COMPILER_DIALECT_HIP_IR_HIPBUFFERIZE_H

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {
namespace hip {

/// Bufferize a DPS HIP compute op: replace tensor operands with their buffers,
/// recreate the op with no result types, then replace each tensor result with
/// the buffer that was tied to the corresponding DPS init operand.
template <typename OpTy, typename ModelTy>
static mlir::LogicalResult hipDpsBufferize(
    mlir::Operation *op, mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  auto dstOp = mlir::cast<mlir::DestinationStyleOpInterface>(op);

  // Build the new operand list: bufferize tensor operands, keep the rest.
  mlir::SmallVector<mlir::Value> newOperands;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (mlir::isa<mlir::TensorType>(operand.get().getType())) {
      mlir::FailureOr<mlir::Value> buf =
          mlir::bufferization::getBuffer(rewriter, operand.get(), options,
                                         state);
      if (mlir::failed(buf))
        return mlir::failure();
      newOperands.push_back(*buf);
    } else {
      newOperands.push_back(operand.get());
    }
  }

  // Recreate the op without tensor result types.
  mlir::Operation *newOp =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), newOperands,
                      /*resultTypes=*/mlir::TypeRange{}, op->getAttrs());
  (void)newOp;

  // Each tensor result aliases its tied DPS init operand buffer.
  mlir::SmallVector<mlir::Value> replacements;
  for (mlir::OpResult result : op->getResults()) {
    mlir::OpOperand *initOperand = dstOp.getTiedOpOperand(result);
    mlir::FailureOr<mlir::Value> buf =
        mlir::bufferization::getBuffer(rewriter, initOperand->get(), options,
                                       state);
    if (mlir::failed(buf))
      return mlir::failure();
    replacements.push_back(*buf);
  }
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                     replacements);
  return mlir::success();
}

/// External bufferization model for destination-passing-style HIP compute ops.
///
/// Provides the full DPS bufferization behaviour:
///   - bufferizesToMemoryRead  = true  for every non-init operand
///   - bufferizesToMemoryWrite = true  for every DPS init operand
///   - getAliasingValues       = Equivalent between each DPS init and result
///   - bufferize()             = replace tensor operands with buffers,
///                               recreate op with TypeRange{}, replace results
///
/// Used for all 16 HIP compute ops.
template <typename OpTy>
struct HipDstBufferizableModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          HipDstBufferizableModel<OpTy>, OpTy> {
  mlir::LogicalResult bufferize(
      mlir::Operation *op, mlir::RewriterBase &rewriter,
      const mlir::bufferization::BufferizationOptions &options,
      mlir::bufferization::BufferizationState &state) const {
    return hipDpsBufferize<OpTy, HipDstBufferizableModel<OpTy>>(op, rewriter,
                                                                 options, state);
  }
};

/// Specialised bufferization model for elementwise HIP ops (relu, cast).
///
/// Identical semantics to HipDstBufferizableModel but kept as a separate
/// template so that op-specific aliasing logic can be added incrementally.
template <typename OpTy>
struct HipElementwiseBufferizableModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          HipElementwiseBufferizableModel<OpTy>, OpTy> {
  mlir::LogicalResult bufferize(
      mlir::Operation *op, mlir::RewriterBase &rewriter,
      const mlir::bufferization::BufferizationOptions &options,
      mlir::bufferization::BufferizationState &state) const {
    return hipDpsBufferize<OpTy, HipElementwiseBufferizableModel<OpTy>>(
        op, rewriter, options, state);
  }
};

/// Register HipDstBufferizableModel for all 14 non-elementwise compute ops
/// and HipElementwiseBufferizableModel for relu and cast.
///
/// Call this before the MLIRContext loads dialects so that the external
/// models are attached to ops during dialect initialisation:
///
///   DialectRegistry registry;
///   registerAllDialects(registry);
///   registerHipBufferizableOpInterfaceModels(registry);
///   context.appendDialectRegistry(registry);
///
void registerHipBufferizableOpInterfaceModels(DialectRegistry &registry);

} // namespace hip
} // namespace mlir

#endif // hip_COMPILER_DIALECT_HIP_IR_HIPBUFFERIZE_H
