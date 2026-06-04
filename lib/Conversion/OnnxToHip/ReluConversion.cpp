/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReluConversion.cpp - onnx.Relu -> hip.max -------------------------===//
//
// Relu(x) = max(0, x). We lower it as a single `hip.max` against a 0-D
// `onnx.Constant` zero (which broadcasts against `x`). Emitting the HIP
// dialect op directly — not `onnx.Max` — is required because the
// convert-onnx-to-hip pass runs with
//   GreedyRewriteStrictness::ExistingOps
// so any onnx.* op we synthesize would survive past the pass and trip
// "op was not bufferized" downstream.
//
//   Before:
//     %y = "onnx.Relu"(%x) : (tensor<...xT>) -> tensor<...xT>
//
//   After:
//     %zero = "onnx.Constant"() {value = dense<0> : tensor<T>} : () -> tensor<T>
//     %init = tensor.empty(...) : tensor<...xT>
//     %y    = hip.max(%ctx) ins(%x, %zero : ..., tensor<T>) outs(%init : ...)
//
// `onnx.Constant` is intentionally retained — it is folded / handled by the
// generic constant-handling path and is the canonical way for an ONNX
// converter to introduce a literal at this stage.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

/// Build a 0-D `arith.constant` carrying a zero value of `elemType`.
///
/// We emit `arith.constant` (not `onnx.Constant`) because `convert-onnx-to-hip`
/// runs `lowerOnnxConstants` BEFORE `convertComputeOps`. Any `onnx.Constant` we
/// synthesize from a compute-op pattern is therefore never lowered and trips
/// "op was not bufferized" downstream.
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
  return mlir::arith::ConstantOp::create(rewriter, loc, valueAttr).getResult();
}

struct ReluToHipMax : public mlir::RewritePattern {
  ReluToHipMax(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Relu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Relu expects 1 operand and 1 result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "onnx.Relu expects ranked tensor");

    mlir::Value zero = buildZeroScalar(rewriter, loc, xType.getElementType());

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    // Init tensor for DPS output. Use `x` as the shape source so dynamic
    // dims of the result are tied to the same SSA values as `x` (the scalar
    // zero has rank 0 and carries no dim info).
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

    auto maxOp = mlir::hip::MaxOp::create(rewriter, loc, resultType, context,
                                          x, zero, init);
    rewriter.replaceOp(op, maxOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateReluConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ReluToHipMax>(ctx);
}

} // namespace hip
} // namespace mlir
