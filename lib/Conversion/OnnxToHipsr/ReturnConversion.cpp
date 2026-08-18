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
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

struct ReturnToFunc : public ::mlir::OpRewritePattern<::mlir::onnx::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::ReturnOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::ReturnOp>(op, op.getOperands());
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
