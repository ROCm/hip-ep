/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LeakyReluConversion.cpp - Decompose onnx.LeakyRelu ----------------===//
//
// LeakyRelu(x, alpha) = x if x >= 0 else alpha*x. Lowered without a new
// kernel via Mul + Less + Where:
//
//   Before:
//     %y = "onnx.LeakyRelu"(%x) {alpha = 0.1 : f32}
//          : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %zero   = "onnx.Constant"() {value = dense<0.0> : tensor<T>}
//     %alpha  = "onnx.Constant"() {value = dense<alpha> : tensor<T>}
//     %alphaX = "onnx.Mul"(%alpha, %x)
//     %cond   = "onnx.Less"(%x, %zero)
//     %y      = "onnx.Where"(%cond, %alphaX, %x)
//
// Type T is one of bf16/f16/f32/f64 per the ONNX spec.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace hip {
namespace {

static mlir::Value buildFloatScalar(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::FloatType ft,
                                    double value) {
  auto scalarType = mlir::RankedTensorType::get({}, ft);
  llvm::APFloat apf(value);
  bool losesInfo = false;
  apf.convert(ft.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
              &losesInfo);
  auto valueAttr = mlir::DenseElementsAttr::get(scalarType, apf);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value", valueAttr);
  return rewriter.create(state)->getResult(0);
}

struct LeakyReluDecompose : public mlir::RewritePattern {
  LeakyReluDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LeakyRelu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.LeakyRelu expects 1 operand and 1 result");

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "ranked tensor required");
    auto ft = mlir::dyn_cast<mlir::FloatType>(xType.getElementType());
    if (!ft)
      return rewriter.notifyMatchFailure(op, "float element type required");

    double alphaVal = 0.01;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alphaVal = attr.getValueAsDouble();

    mlir::Value zero = buildFloatScalar(rewriter, loc, ft, 0.0);
    mlir::Value alpha = buildFloatScalar(rewriter, loc, ft, alphaVal);

    mlir::OperationState mulState(loc, "onnx.Mul");
    mulState.addOperands({alpha, x});
    mulState.addTypes(xType);
    mlir::Value alphaX = rewriter.create(mulState)->getResult(0);

    auto condType =
        mlir::RankedTensorType::get(xType.getShape(), rewriter.getI1Type());
    mlir::OperationState lessState(loc, "onnx.Less");
    lessState.addOperands({x, zero});
    lessState.addTypes(condType);
    mlir::Value cond = rewriter.create(lessState)->getResult(0);

    mlir::OperationState whereState(loc, "onnx.Where");
    whereState.addOperands({cond, alphaX, x});
    whereState.addTypes(op->getResult(0).getType());
    mlir::Operation *whereOp = rewriter.create(whereState);
    rewriter.replaceOp(op, whereOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateLeakyReluConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<LeakyReluDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
