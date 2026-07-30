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

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {
namespace hipsr {

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

/// Maps a data-graph input to the matching shape-graph dependency. Function
/// inputs are shared roots. A hipsr DPS result maps to its placeholder init.
inline ::mlir::FailureOr<::mlir::Value>
getPlaceholderDependency(::mlir::Value input, ::mlir::Operation *op,
                         ::mlir::PatternRewriter &rewriter) {
  if (::mlir::isa<::mlir::BlockArgument>(input)) {
    return input;
  }

  auto result = ::mlir::dyn_cast<::mlir::OpResult>(input);
  if (!result) {
    return rewriter.notifyMatchFailure(op,
                                       "input has no shape-graph dependency");
  }
  auto dpsProducer =
      ::mlir::dyn_cast<::mlir::DestinationStyleOpInterface>(result.getOwner());
  if (!dpsProducer || result.getOwner()->getName().getDialectNamespace() !=
                          HipsrDialect::getDialectNamespace()) {
    return rewriter.notifyMatchFailure(
        op, "input is not a function argument or hipsr DPS result");
  }

  ::mlir::Value init = dpsProducer.getTiedOpOperand(result)->get();
  if (!init.getDefiningOp<::mlir::hipsr::PlaceholderOp>()) {
    return rewriter.notifyMatchFailure(
        op, "hipsr DPS input producer has no placeholder init");
  }
  return init;
}

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPSR_UTILS_H
