/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReturnConversion.cpp - Convert onnx.Return to func.return ----------===//
//
// onnx.Return carries graph structure rather than a computation, so it lowers
// to func.return instead of to a hipsr operation.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

struct ReturnToFunc : public ::mlir::RewritePattern {
  ReturnToFunc(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Return", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::ReturnOp>(op, op->getOperands());
    return ::mlir::success();
  }
};

} // namespace

void populateReturnConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx) {
  patterns.add<ReturnToFunc>(ctx);
}

} // namespace hipsr
} // namespace mlir
