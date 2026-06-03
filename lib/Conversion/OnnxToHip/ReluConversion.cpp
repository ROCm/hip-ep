/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReluConversion.cpp - Decompose onnx.Relu into Less + Where --------===//
//
// Relu(x) = max(0, x). MorphiZen does not have an elementwise-max ONNX
// converter, but Less + Where together express the same selection without
// a new kernel:
//
//   Before:
//     %y = "onnx.Relu"(%x) : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %zero = "onnx.Constant"() {value = dense<0> : tensor<T>} : () -> tensor<T>
//     %cond = "onnx.Less"(%x, %zero) : (tensor<...xT>, tensor<T>) -> tensor<...xi1>
//     %y    = "onnx.Where"(%cond, %zero, %x)
//             : (tensor<...xi1>, tensor<T>, tensor<...xT>) -> tensor<...xT>
//
// The scalar zero broadcasts against x; Less and Where converters already
// support the broadcast pattern.
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

    mlir::Type elemType = xType.getElementType();
    mlir::Value zero = buildZeroScalar(rewriter, loc, elemType);

    auto i1Type = rewriter.getI1Type();
    auto condType = mlir::RankedTensorType::get(xType.getShape(), i1Type);

    mlir::OperationState lessState(loc, "onnx.Less");
    lessState.addOperands({x, zero});
    lessState.addTypes(condType);
    mlir::Value cond = rewriter.create(lessState)->getResult(0);

    mlir::OperationState whereState(loc, "onnx.Where");
    whereState.addOperands({cond, zero, x});
    whereState.addTypes(op->getResult(0).getType());
    mlir::Operation *whereOp = rewriter.create(whereState);
    rewriter.replaceOp(op, whereOp->getResult(0));
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
