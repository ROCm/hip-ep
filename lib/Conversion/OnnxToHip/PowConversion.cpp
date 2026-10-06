/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PowConversion.cpp - onnx.Pow -> hip.pow ---------------------------===//
//
// Constant scalar exponents that PowDecompose cannot express as Mul, Sqrt, or
// Reciprocal become hip.pow. The exponent is a compile-time f64 attribute.
// A non-constant exponent is left unmatched.
//
// PowDecompose runs in pre-lowering, before lowerOnnxConstants. This pattern
// runs in convertComputeOps, so the exponent is a hip.constant (or still an
// arith.constant in hand-written tests), optionally behind a cast.
//
//   Before:
//     %e = "onnx.Constant"() {value = dense<2.2> : tensor<f32>} : () ->
//     tensor<f32> %y = "onnx.Pow"(%x, %e) : (tensor<4x8xf32>, tensor<f32>) ->
//     tensor<4x8xf32>
//   After:
//     %init = tensor.empty() : tensor<4x8xf32>
//     %y = hip.pow(%ctx) ins(%x : tensor<4x8xf32>) outs(%init :
//     tensor<4x8xf32>)
//            {exponent = 2.2 : f64}
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>
#include <optional>

namespace mlir {
namespace hip {
namespace {

static mlir::Value unwrapExponentCast(mlir::Value v) {
  for (int i = 0; i < 4; ++i) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      break;
    llvm::StringRef name = def->getName().getStringRef();
    if ((name == "onnx.Cast" || name == "onnx.CastLike") &&
        def->getNumOperands() >= 1) {
      v = def->getOperand(0);
      continue;
    }
    if (mlir::isa<mlir::hip::CastOp>(def) && def->getNumOperands() >= 2) {
      v = def->getOperand(1);
      continue;
    }
    break;
  }
  return v;
}

static std::optional<double> readScalarConstant(mlir::Value v) {
  mlir::Operation *def = unwrapExponentCast(v).getDefiningOp();
  if (!def)
    return std::nullopt;

  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (auto attr = def->getAttr("value"))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense || dense.getNumElements() != 1)
    return std::nullopt;

  mlir::Type et = dense.getElementType();
  if (et.isF32())
    return static_cast<double>(*dense.getValues<float>().begin());
  if (et.isF64())
    return *dense.getValues<double>().begin();
  if (et.isF16() || et.isBF16())
    return (*dense.getValues<llvm::APFloat>().begin()).convertToDouble();
  if (et.isIntOrIndex())
    return static_cast<double>(
        (*dense.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

/// Round through the exponent tensor's element type so a cast that ONNX would
/// apply before Pow is honored.
static std::optional<double> readExponent(mlir::Value expVal) {
  std::optional<double> raw = readScalarConstant(expVal);
  if (!raw || !std::isfinite(*raw))
    return std::nullopt;
  auto shaped = mlir::dyn_cast<mlir::RankedTensorType>(expVal.getType());
  if (!shaped)
    return raw;
  auto ft = mlir::dyn_cast<mlir::FloatType>(shaped.getElementType());
  if (!ft)
    return raw;
  llvm::APFloat ap(*raw);
  bool losesInfo = false;
  ap.convert(ft.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
             &losesInfo);
  return ap.convertToDouble();
}

static bool isPowFloatType(mlir::Type type) {
  auto ft = mlir::dyn_cast<mlir::FloatType>(type);
  return ft && (ft.isF16() || ft.isF32() || ft.isF64() || ft.isBF16());
}

struct PowToHip : public mlir::RewritePattern {
  PowToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pow", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "onnx.Pow expects 2 operands");

    mlir::Value base = op->getOperand(0);
    mlir::Value exponent = op->getOperand(1);
    auto baseType = mlir::dyn_cast<mlir::RankedTensorType>(base.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!baseType || !resultType)
      return rewriter.notifyMatchFailure(op, "ranked tensor required");
    if (baseType.getElementType() != resultType.getElementType() ||
        !isPowFloatType(baseType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "Pow runtime supports matching f16, f32, bf16, and f64");
    if (baseType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(op, "Pow result rank must match base");
    for (int64_t i : llvm::seq<int64_t>(baseType.getRank())) {
      if (!baseType.isDynamicDim(i) && !resultType.isDynamicDim(i) &&
          baseType.getDimSize(i) != resultType.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "Pow result shape must match the base");
    }

    std::optional<double> exp = readExponent(exponent);
    if (!exp)
      return rewriter.notifyMatchFailure(
          op, "Pow exponent is not a finite compile-time scalar");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, base);
    auto hipOp =
        mlir::hip::PowOp::create(rewriter, loc, resultType, *ctxOrFailure, base,
                                 init, rewriter.getF64FloatAttr(*exp));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populatePowConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<PowToHip>(ctx);
}

} // namespace hip
} // namespace mlir
