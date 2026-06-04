/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LeakyReluConversion.cpp - onnx.LeakyRelu -> hip.{mul,less,where} --===//
//
// LeakyRelu(x, alpha) = x if x >= 0 else alpha*x. Decomposed without a new
// kernel into Mul + Less + Where — but emitted as HIP-dialect ops directly,
// not onnx.* ops, because the convert-onnx-to-hip driver runs with
// GreedyRewriteStrictness::ExistingOps and would not pick up newly-created
// onnx.* nodes.
//
//   Before:
//     %y = "onnx.LeakyRelu"(%x) {alpha = 0.1 : f32}
//          : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %zero   = "onnx.Constant"() {value = dense<0.0> : tensor<T>}
//     %alpha  = "onnx.Constant"() {value = dense<alpha> : tensor<T>}
//     %i0     = tensor.empty(...) : tensor<...xT>
//     %alphaX = hip.mul(%ctx) ins(%alpha, %x : ...) outs(%i0 : ...)
//     %ci     = tensor.empty(...) : tensor<...xi1>
//     %cond   = hip.less(%ctx) ins(%x, %zero : ...) outs(%ci : tensor<...xi1>)
//     %iy     = tensor.empty(...) : tensor<...xT>
//     %y      = hip.where(%ctx) ins(%cond, %alphaX, %x : ...) outs(%iy : ...)
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

struct LeakyReluToHip : public mlir::RewritePattern {
  LeakyReluToHip(mlir::MLIRContext *ctx)
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

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    double alphaVal = 0.01;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alphaVal = attr.getValueAsDouble();

    mlir::Value zero = buildFloatScalar(rewriter, loc, ft, 0.0);
    mlir::Value alpha = buildFloatScalar(rewriter, loc, ft, alphaVal);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // hip.mul(alpha, x) -> alphaX. `x` is the rank source for the init.
    mlir::Value mulInit = createEmptyTensor(rewriter, loc, resultType, x);
    mlir::Value alphaX = mlir::hip::MulOp::create(rewriter, loc, resultType,
                                                  context, alpha, x, mulInit)
                             ->getResult(0);

    // hip.less(x, zero) -> cond (i1, same shape as x).
    auto condType =
        mlir::RankedTensorType::get(xType.getShape(), rewriter.getI1Type());
    mlir::Value condInit = createEmptyTensor(rewriter, loc, condType, x);
    mlir::Value cond = mlir::hip::LessOp::create(rewriter, loc, condType,
                                                 context, x, zero, condInit)
                           ->getResult(0);

    // hip.where(cond, alphaX, x) -> y.
    mlir::Value whereInit = createEmptyTensor(rewriter, loc, resultType, x);
    mlir::Value y = mlir::hip::WhereOp::create(rewriter, loc, resultType,
                                               context, cond, alphaX, x,
                                               whereInit)
                        ->getResult(0);

    rewriter.replaceOp(op, y);
    return mlir::success();
  }
};

} // namespace

void populateLeakyReluConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<LeakyReluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
