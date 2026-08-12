/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsrUtils.h - Shared helpers for ONNX-to-hipsr patterns ----===//
//
// Shared utility functions used by the per-operator ONNX-to-hipsr conversion
// files. Private to lib/Conversion/OnnxToHipsr (not an installed public
// header), mirroring lib/Conversion/OnnxToHip/OnnxToHipUtils.h.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIPSR_UTILS_H
#define HIP_CONVERSION_ONNXTOHIPSR_UTILS_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

// Force change memoryspace to space
inline ::mlir::RankedTensorType tensorTypeInSpace(::mlir::RankedTensorType type,
                                                  MemorySpace space) {
  return type.cloneWithEncoding(
      ::mlir::hipsr::MemorySpaceAttr::get(type.getContext(), space));
}

/// Creates `compute`'s entry block and leaves `builder` inserting at its
/// start. The ODS builder only reserves the region, so every conversion that
/// emits a compute has to fill it.
inline ::mlir::Block &createComputeBodyBlock(::mlir::OpBuilder &builder,
                                             ComputeOp compute) {
  // The body's arguments are the operands: ctx, then the inputs, then the
  // outputs.
  ::mlir::TypeRange argTypes = compute->getOperandTypes();
  ::llvm::SmallVector<::mlir::Location> argLocs(argTypes.size(),
                                                compute.getLoc());
  ::mlir::Block *block =
      builder.createBlock(&compute.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(block);
  return *block;
}

/// Returns `compute`'s entry-block argument for input `index`, skipping the
/// leading `ctx` argument.
inline ::mlir::Value computeBodyInput(::mlir::Block &body, unsigned index) {
  return body.getArgument(1 + index);
}

/// Terminates a placeholder's shape region with `extents`. The region yields
/// one `!shape.shape` however it was computed, so only the extents differ
/// between the conversions that fill one.
inline void yieldShapeFromExtents(::mlir::OpBuilder &builder,
                                  ::mlir::Location loc,
                                  ::mlir::ValueRange extents) {
  ::mlir::Value shape = ::mlir::shape::FromExtentsOp::create(
      builder, loc, ::mlir::shape::ShapeType::get(builder.getContext()),
      extents);
  ShapeYieldOp::create(builder, loc, ::mlir::ValueRange{shape});
}

/// Gets the `!hipsr.context` from function argument 0. The ONNX phase adds it
/// as the enclosing function's first argument, so every hipsr op can thread it
/// as its own first operand (mirrors hip's getContextArg). Returns failure (so
/// the calling pattern bails) if the op is not inside a function body, the
/// function has no arguments, or arg 0 is not `!hipsr.context`.
inline ::mlir::FailureOr<::mlir::Value>
getHipsrContextArg(::mlir::Operation *op, ::mlir::PatternRewriter &rewriter) {
  auto funcOp = op->getParentOfType<::mlir::func::FuncOp>();
  if (!funcOp || funcOp.getBody().empty()) {
    return rewriter.notifyMatchFailure(op, "not inside a function body");
  }
  ::mlir::Block &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0) {
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  }
  ::mlir::Value ctx = entry.getArgument(0);
  if (!::mlir::isa<::mlir::hipsr::ContextType>(ctx.getType())) {
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hipsr.context");
  }
  return ctx;
}

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_UTILS_H
