/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gemm_to_matmul.cpp - hip.gemm to hip.matmul ------------------------===//
//
//   Before:
//     %y = hip.gemm(%ctx) ins(%a, %w, %c : tensor<64x128xT>, tensor<32x128xT>,
//                                          tensor<32xT>)
//                         outs(%e : tensor<64x32xT>) {transB = 1 : i64}
//   After:
//     %y = hip.matmul(%ctx) ins(%a, %w, %c : ...) outs(%e : tensor<64x32xT>)
//                           {transB = 1 : i64}
//
// Type-agnostic: float and quant-typed gemms convert alike, so the order
// against the Q/DQ folds does not matter. A weight with per-channel or packed
// 4-bit Q/DQ keeps hip.gemm: only QGemmFusion.pdll handles those.
//===----------------------------------------------------------------------===//

#include "gemm_to_matmul.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {
namespace {

using namespace mlir;
using namespace mlir::hip;

/// `w` is dequantized per channel or from packed 4-bit storage.
bool isNonUniformWeight(Value w) {
  auto dq = w.getDefiningOp<DequantizeLinearOp>();
  if (!dq)
    return false;
  if (dq->hasAttr("packed_int4"))
    return true;
  auto scale = dq.getScale().getDefiningOp<ConstantOp>();
  auto payload =
      scale ? dyn_cast_or_null<DenseElementsAttr>(scale->getAttr("value"))
            : DenseElementsAttr();
  return !payload || !payload.isSplat();
}

struct GemmToMatmul : public OpRewritePattern<GemmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp gemm,
                                PatternRewriter &rewriter) const override {
    if (gemm->getNumResults() != 1 || !gemm.getAlpha().isExactlyValue(1.0))
      return failure();
    if (isNonUniformWeight(gemm.getInputB()))
      return rewriter.notifyMatchFailure(gemm, "per-channel or packed weight");

    auto yType = cast<RankedTensorType>(gemm->getResult(0).getType());
    Value bias = gemm.getInputC();
    if (bias) {
      // Only a row bias is a matmul bias; beta scales it.
      auto cType = cast<RankedTensorType>(bias.getType());
      int64_t n = yType.getShape().back();
      bool rowBias = cType.hasStaticShape() && !ShapedType::isDynamic(n) &&
                     (cType.getRank() == 1 ||
                      (cType.getRank() == 2 && cType.getDimSize(0) == 1)) &&
                     cType.getShape().back() == n;
      if (!rowBias || !gemm.getBeta().isExactlyValue(1.0))
        return rewriter.notifyMatchFailure(gemm, "not a row bias");
      if (cType.getRank() == 2)
        bias = tensor::CollapseShapeOp::create(
            rewriter, gemm.getLoc(), bias,
            SmallVector<ReassociationIndices>{{0, 1}});
    }

    auto matmul = MatmulOp::create(
        rewriter, gemm.getLoc(), TypeRange{yType}, gemm.getCtx(),
        gemm.getInputA(), gemm.getInputB(), bias, gemm.getOutput(),
        gemm.getTransAAttr(), gemm.getTransBAttr(), gemm.getQuantTypesAttr());
    rewriter.replaceOp(gemm, matmul->getResults());
    return success();
  }
};

} // namespace

void populateGemmToMatmulPattern(RewritePatternSet &patterns,
                                 PatternBenefit benefit) {
  patterns.add<GemmToMatmul>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
