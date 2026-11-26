/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_bias_fusion.cpp - fold a bias add into MatMulNBits ----===//
//
// Before:
//   %y = hip.matmul_nbits(%ctx) ins(%a, %b, %scales) outs(%init)
//   %z = hip.add(%ctx) ins(%y, %bias) outs(%init2)
// After:
//   %z = hip.matmul_nbits(%ctx) ins(%a, %b, %scales) bias(%bias) outs(%init2)
//
// Every MatMulNBits kernel adds the bias in f32 before the output conversion,
// so the fused result is the unfused one without the rounding of the
// intermediate product to the output type.
//
// Safety: the matmul result must have the add as its single value use (a
// tensor.dim query is not a value use; it is retargeted to the matmul's init),
// the add result must have the matmul's type so the add broadcasts nothing
// but the bias, and the bias must be the [N] vector the op's bias operand
// describes, in the output element type the kernels read it as.
//
//===----------------------------------------------------------------------===//

#include "matmul_nbits_bias_fusion.hpp"

#include "fusion_utils.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <utility>

namespace hip {
namespace fusion_transform {
namespace {

struct MatMulNBitsBiasFusion : public mlir::OpRewritePattern<mlir::hip::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::AddOp add,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::hip::MatMulNBitsOp matmul;
    mlir::Value bias;
    for (auto [product, other] : {std::pair(add.getLhs(), add.getRhs()),
                                  std::pair(add.getRhs(), add.getLhs())}) {
      auto candidate = product.getDefiningOp<mlir::hip::MatMulNBitsOp>();
      if (candidate && hasSingleValueUse(product, add)) {
        matmul = candidate;
        bias = other;
        break;
      }
    }
    if (!matmul)
      return rewriter.notifyMatchFailure(add, "no single-use matmul_nbits");
    if (matmul.getBias())
      return rewriter.notifyMatchFailure(add, "matmul_nbits has a bias");
    if (matmul->getNumResults() != 1)
      return rewriter.notifyMatchFailure(add,
                                         "matmul_nbits is not tensor form");

    auto productType =
        mlir::dyn_cast<mlir::RankedTensorType>(matmul->getResult(0).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(add->getResult(0).getType());
    if (!productType || productType != resultType)
      return rewriter.notifyMatchFailure(add, "add broadcasts the product");

    auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
    if (!biasType || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != static_cast<int64_t>(matmul.getN()) ||
        biasType.getElementType() != productType.getElementType())
      return rewriter.notifyMatchFailure(add, "bias is not an [N] vector");

    // The HIP context is operand 0 of every HIP op.
    if (add.getCtx() != matmul.getHandle())
      return rewriter.notifyMatchFailure(add, "ops do not share a context");

    // A shape query may sit between the matmul and the add, before the fused
    // op exists, so it reads the matmul's init instead: same type, and an
    // operand of the matmul, so it dominates every use of the product.
    mlir::Value init = matmul.getOutput();
    retargetShapeQueries(rewriter, matmul->getResult(0), init);
    auto fused = mlir::hip::MatMulNBitsOp::create(
        rewriter, add.getLoc(), mlir::TypeRange{resultType}, add.getCtx(),
        matmul.getA(), matmul.getB(), matmul.getScales(),
        matmul.getZeroPoints(), matmul.getGIdx(), bias, add.getOutput(),
        matmul.getKAttr(), matmul.getNAttr(), matmul.getBitsAttr(),
        matmul.getBlockSizeAttr(), matmul.getAccuracyLevelAttr(),
        matmul.getZpElemSizeAttr(), matmul.getScaleElemSizeAttr());
    rewriter.replaceOp(add, fused->getResults());
    rewriter.eraseOp(matmul);
    // The matmul's own init is dead unless a retargeted shape query reads it.
    if (mlir::Operation *def = init.getDefiningOp())
      if (mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

} // namespace

void populateMatMulNBitsBiasFusionPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit) {
  patterns.add<MatMulNBitsBiasFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
