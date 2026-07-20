/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

/// onnx.MatMul -> hipsr.matmul. The shape region is left empty here; a later
/// pass populates every op's shape region uniformly (see populateShapeRegion).
///
/// The ONNX phase prepends the per-session `!hipsr.context` as operand 0 of
/// every onnx op, so the conversion reads it straight off this op's first
/// operand (not from the enclosing function) and threads it onto hipsr.matmul.
///
/// Before:
///   %0 = "onnx.MatMul"(%ctx, %A, %B)
///          : (!hipsr.context, tensor<?x?xf16>, tensor<?x?xf16>)
///          -> tensor<?x?xf16>
/// After:
///   %init = hipsr.placeholder : tensor<?x?xf16>
///   %0 = hipsr.matmul(%ctx) ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
///                           outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
struct MatMulToHipsr : public ::mlir::RewritePattern {
  MatMulToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Matching is by name on an (unregistered) ONNX op, so guard the shape.
    // With the ONNX phase's ctx-prepend, a threaded onnx.MatMul carries
    // ctx + the two matrix inputs (three operands), single-result.
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected a context operand, two matrix operands, one result");

    // %ctx is the op's first operand (prepended in the ONNX phase).
    ::mlir::Value ctx = op->getOperand(0);
    if (!::mlir::isa<::mlir::hipsr::ContextType>(ctx.getType()))
      return rewriter.notifyMatchFailure(op, "operand 0 is not !hipsr.context");

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value a = op->getOperand(1);
    ::mlir::Value b = op->getOperand(2);
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    // DPS init: a hipsr.placeholder mirroring the result type. That is all the
    // DPS verifier needs, and it avoids computing the output shape here (no
    // tensor.empty + tensor.dim).
    ::mlir::Value init =
        rewriter.create<PlaceholderOp>(loc, ::mlir::TypeRange{resultType})
            .getResult(0);

    // Shape region left empty (zero blocks); a later pass populates it.
    auto matmulOp = rewriter.create<MatMulOp>(
        loc, ::mlir::TypeRange{resultType}, ctx, a, b, init);

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateMatMulConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx) {
  patterns.add<MatMulToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
