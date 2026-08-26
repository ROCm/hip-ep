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
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {

// Force change memoryspace to space
inline ::mlir::RankedTensorType tensorTypeInSpace(::mlir::RankedTensorType type,
                                                  MemorySpace space) {
  return type.cloneWithEncoding(
      ::mlir::hipsr::MemorySpaceAttr::get(type.getContext(), space));
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

/// Reads `src` back to the host: builds the host destination placeholder, fills
/// its shape region, and returns the `hipsr.copy_d2h` that writes it.
///
/// A copy only changes memory space, so the destination takes the source shape
/// and the region just forwards it. Filling it here keeps a readback to one
/// call, and the shape it yields is the one `AllShapesMatch` already pins.
inline CopyD2HOp createCopyD2H(::mlir::OpBuilder &builder, ::mlir::Location loc,
                               ::mlir::Value ctx, ::mlir::Value src) {
  auto hostType = tensorTypeInSpace(
      ::mlir::cast<::mlir::RankedTensorType>(src.getType()), MemorySpace::Host);
  auto init =
      PlaceholderOp::create(builder, loc, ::mlir::TypeRange{hostType}, ctx,
                            ::mlir::ValueRange{src}, PlaceholderType::Normal);
  {
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    ::mlir::Block &block = createPlaceholderShapeBlock(builder, init);
    builder.setInsertionPointToStart(&block);
    ShapeYieldOp::create(
        builder, loc,
        ::mlir::ValueRange{PlaceholderShapeRegionArgs{block}.in(0)});
  }
  return CopyD2HOp::create(builder, loc, ::mlir::TypeRange{hostType}, ctx, src,
                           init.getResult(0));
}

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_UTILS_H
