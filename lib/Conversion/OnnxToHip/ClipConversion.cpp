/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ClipConversion.cpp - onnx.Clip -> hip.clip ------------------------===//
//
// ONNX Clip (opset 11+) takes the input plus two optional 0-rank scalar
// operands `min` and `max`. When either is absent, the spec defaults it to
// `numeric_limits::lowest()` / `numeric_limits::max()` of the element type.
//
// We normalize at compile time: if `min` or `max` is missing, synthesize an
// `onnx.Constant` scalar of the same dtype carrying the default. The
// `hip.clip` op then always has both bounds, which keeps the runtime kernel
// simple.
//
//   Before:
//     %y = "onnx.Clip"(%x)                : (tensor<...xT>) -> tensor<...xT>
//     %y = "onnx.Clip"(%x, %lo)           : (tensor<...xT>, tensor<T>)
//                                                 -> tensor<...xT>
//     %y = "onnx.Clip"(%x, %lo, %hi)      : (tensor<...xT>, tensor<T>,
//                                            tensor<T>) -> tensor<...xT>
//
//   After (uniform shape after default synthesis):
//     %y = hip.clip(%ctx) ins(%x, %lo, %hi : ...) outs(%init : ...)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

/// Build a 0-D `onnx.Constant` holding the dtype's lowest/highest finite
/// value. Used to fill in omitted `min` / `max` operands of `onnx.Clip`.
static mlir::Value buildBoundScalar(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Type elemType,
                                    bool isLower) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);
  mlir::DenseElementsAttr valueAttr;
  if (auto ft = mlir::dyn_cast<mlir::FloatType>(elemType)) {
    llvm::APFloat apf =
        llvm::APFloat::getLargest(ft.getFloatSemantics(), /*Negative=*/isLower);
    valueAttr = mlir::DenseElementsAttr::get(scalarType, apf);
  } else {
    auto it = mlir::cast<mlir::IntegerType>(elemType);
    unsigned bw = it.getWidth();
    llvm::APInt apv =
        isLower
            ? (it.isUnsigned() ? llvm::APInt::getZero(bw)
                               : llvm::APInt::getSignedMinValue(bw))
            : (it.isUnsigned() ? llvm::APInt::getMaxValue(bw)
                               : llvm::APInt::getSignedMaxValue(bw));
    valueAttr = mlir::DenseElementsAttr::get(scalarType, apv);
  }
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value", valueAttr);
  return rewriter.create(state)->getResult(0);
}

/// `onnx.Clip` may carry an `onnx.NoValue` as a placeholder for an omitted
/// optional operand. Treat such operands as missing.
static bool isNoValue(mlir::Value v) {
  if (!v)
    return true;
  if (mlir::Operation *def = v.getDefiningOp())
    return def->getName().getStringRef() == "onnx.NoValue";
  return false;
}

struct ClipToHip : public mlir::RewritePattern {
  ClipToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Clip", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "onnx.Clip expects 1 result");
    unsigned n = op->getNumOperands();
    if (n < 1 || n > 3)
      return rewriter.notifyMatchFailure(op, "onnx.Clip expects 1..3 operands");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "ranked tensor required");
    mlir::Type elemType = xType.getElementType();

    mlir::Value minV = (n >= 2 && !isNoValue(op->getOperand(1)))
                           ? op->getOperand(1)
                           : buildBoundScalar(rewriter, loc, elemType, true);
    mlir::Value maxV = (n >= 3 && !isNoValue(op->getOperand(2)))
                           ? op->getOperand(2)
                           : buildBoundScalar(rewriter, loc, elemType, false);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

    auto clipOp = mlir::hip::ClipOp::create(rewriter, loc, resultType, context,
                                            x, minV, maxV, init);
    rewriter.replaceOp(op, clipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateClipConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ClipToHip>(ctx);
}

} // namespace hip
} // namespace mlir
