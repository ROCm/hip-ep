/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReluConversion.cpp - Decompose onnx.Relu into onnx.Max ------------===//
//
// Relu(x) = max(0, x). We emit `onnx.Max(x, 0)` so that the same MaxToHip
// conversion (which delegates to MIOpen's miopenOpTensor) handles it. The
// scalar zero broadcasts against `x`; the Max pattern supports broadcast.
//
//   Before:
//     %y = "onnx.Relu"(%x) : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %zero = "onnx.Constant"() {value = dense<0> : tensor<T>} : () ->
//     tensor<T> %y    = "onnx.Max"(%x, %zero) : (tensor<...xT>, tensor<T>)
//                                         -> tensor<...xT>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

/// Build a 0-D `onnx.Constant` carrying a zero value of `elemType`.
static mlir::Value buildZeroScalar(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Type elemType) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);
  mlir::DenseElementsAttr valueAttr;
  if (auto fpType = mlir::dyn_cast<mlir::FloatType>(elemType)) {
    llvm::APFloat zero(fpType.getFloatSemantics(), 0);
    valueAttr = mlir::DenseElementsAttr::get(scalarType, zero);
  } else {
    auto intType = mlir::cast<mlir::IntegerType>(elemType);
    llvm::APInt zero(intType.getWidth(), 0, /*isSigned=*/true);
    valueAttr = mlir::DenseElementsAttr::get(scalarType, zero);
  }
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value", valueAttr);
  return rewriter.create(state)->getResult(0);
}

struct ReluDecompose : public mlir::RewritePattern {
  ReluDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Relu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Relu expects 1 operand and 1 result");

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "onnx.Relu expects ranked tensor");

    mlir::Value zero = buildZeroScalar(rewriter, loc, xType.getElementType());

    mlir::OperationState maxState(loc, "onnx.Max");
    maxState.addOperands({x, zero});
    maxState.addTypes(op->getResult(0).getType());
    mlir::Operation *maxOp = rewriter.create(maxState);
    rewriter.replaceOp(op, maxOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateReluConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ReluDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
