/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_bias_fusion.cpp - hip.matmul + hip.add to matmul with bias --===//
//
//   Before (quantized; the types are on the tensors):
//     %m = hip.matmul(%ctx) ins(%a : tensor<1x196x512xT0>,
//                               %w : tensor<512x1536xT1>)
//                     outs(%e0 : tensor<1x196x1536xT2>) : tensor<...xT2>
//     %y = hip.add(%ctx) ins(%m, %bias : tensor<1x196x1536xT2>,
//                                        tensor<1536xT3>)
//                        outs(%e1 : tensor<1x196x1536xT4>) -> tensor<...xT4>
//   After:
//     %y = hip.matmul(%ctx) ins(%a, %w, %bias : tensor<1x196x512xT0>,
//                                tensor<512x1536xT1>, tensor<1536xT3>)
//                     outs(%e1 : tensor<1x196x1536xT4>) : tensor<...xT4>
//
// The op keeps A's shape: the leading dims fold into the rows at lowering, as
// they do for a batched matmul, so no reshape appears in the IR. That fold is
// only a view when A's rows are outermost, hence transA = 0 for rank > 2, and
// the bias needs a rank-2 weight (hip.matmul's verifier states the same).
//
// The matmul result must have the add as its only use, and the add must read
// it directly: a quantized chain whose matmul and add disagree on the
// intermediate quant type has a requantization between them and is left alone.
//===----------------------------------------------------------------------===//

#include "matmul_bias_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipQuantTypes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace hip {
namespace fusion_transform {
namespace {

using namespace mlir;
using namespace mlir::hip;

/// `bias` broadcasts over every row of an [..., N] result: static, N along its
/// last dim and 1 everywhere else.
bool isRowBias(RankedTensorType bias, RankedTensorType result) {
  int64_t n = result.getShape().back();
  return bias.hasStaticShape() && bias.getRank() >= 1 &&
         bias.getRank() <= result.getRank() && bias.getShape().back() == n &&
         llvm::all_of(bias.getShape().drop_back(),
                      [](int64_t dim) { return dim == 1; });
}

struct MatmulBiasFusion : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp add,
                                PatternRewriter &rewriter) const override {
    if (add->getNumResults() != 1)
      return failure();
    // Addition commutes: the matmul may feed either side.
    for (bool matmulIsLhs : {true, false}) {
      Value matmulValue = matmulIsLhs ? add.getLhs() : add.getRhs();
      Value bias = matmulIsLhs ? add.getRhs() : add.getLhs();
      if (succeeded(fuse(add, matmulValue, bias, rewriter)))
        return success();
    }
    return failure();
  }

private:
  LogicalResult fuse(AddOp add, Value matmulValue, Value bias,
                     PatternRewriter &rewriter) const {
    auto matmul = matmulValue.getDefiningOp<MatmulOp>();
    // Only a plain matmul: one that already has a bias is not fused twice.
    if (!matmul || matmul.getBias() || !matmulValue.hasOneUse())
      return failure();

    auto aType = cast<RankedTensorType>(matmul.getA().getType());
    auto bType = cast<RankedTensorType>(matmul.getB().getType());
    auto matmulType = cast<RankedTensorType>(matmulValue.getType());
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    auto yType = cast<RankedTensorType>(add->getResult(0).getType());

    // The add must not broadcast the matmul result, and the op's verifier
    // rules (rank-2 weight, transA = 0 for rank > 2) must hold.
    if (!biasType || bType.getRank() != 2 || aType.getRank() < 2 ||
        matmulType.getShape() != yType.getShape() ||
        !isRowBias(biasType, yType))
      return failure();
    if (aType.getRank() > 2 && matmul.getTransA() != 0)
      return failure();
    if (!supportsMatmulWithBias(aType.getElementType(), bType.getElementType(),
                                biasType.getElementType(),
                                yType.getElementType()))
      return rewriter.notifyMatchFailure(add, "no kernel for this matmul");

    // A [1, N] bias is the rank-1 [N] row the op takes.
    Value bias1d = bias;
    if (biasType.getRank() > 1)
      bias1d = tensor::CollapseShapeOp::create(
          rewriter, add.getLoc(), bias,
          SmallVector<ReassociationIndices>{
              llvm::to_vector(llvm::seq<int64_t>(0, biasType.getRank()))});

    // The add's init already has the fused result's type and dynamic extents.
    Value fused = MatmulOp::create(
                      rewriter, add.getLoc(), TypeRange{yType}, matmul.getCtx(),
                      matmul.getA(), matmul.getB(), bias1d, add.getOutput(),
                      matmul.getTransAAttr(), matmul.getTransBAttr(),
                      /*quant_types=*/ArrayAttr())
                      ->getResult(0);

    rewriter.replaceOp(add, fused);
    // The matmul's only use was the add. DPS ops are not erased as trivially
    // dead, so do it here.
    rewriter.eraseOp(matmul);
    return success();
  }
};

} // namespace

void populateMatmulBiasFusionPattern(RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<MatmulBiasFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
