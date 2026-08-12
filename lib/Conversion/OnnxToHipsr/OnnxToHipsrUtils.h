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
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

/// Creates `compute`'s entry block and leaves `builder` inserting at its
/// start. The arguments are `ctx`, then the inputs, then the outputs, as
/// hipsr.compute documents; the ODS builder only reserves the region, so a
/// conversion that emits a compute has to build the block and its terminator
/// itself.
inline ::mlir::Block &createComputeBodyBlock(::mlir::OpBuilder &builder,
                                             ComputeOp compute) {
  ::llvm::SmallVector<::mlir::Type> argTypes{compute.getCtx().getType()};
  ::llvm::append_range(argTypes, compute.getInputs().getTypes());
  ::llvm::append_range(argTypes, compute.getOutputs().getTypes());
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

/// Reads extent `axis` of a shape-region `!shape.shape` argument as an `index`.
/// An extent of a `!shape.shape` is a `!shape.size`, which carries the error
/// state that arith cannot express, so it converts before any arithmetic.
inline ::mlir::Value shapeExtentAsIndex(::mlir::OpBuilder &builder,
                                        ::mlir::Location loc,
                                        ::mlir::Value shape, int64_t axis) {
  ::mlir::Value extent =
      ::mlir::shape::GetExtentOp::create(builder, loc, shape, axis);
  return ::mlir::shape::SizeToIndexOp::create(builder, loc,
                                              builder.getIndexType(), extent);
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
