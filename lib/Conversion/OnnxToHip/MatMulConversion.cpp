/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.matmul (custom dense GEMM).
struct MatMulToHip : public mlir::RewritePattern {
  MatMulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  int64_t transA = 0;
  int64_t transB = 0;
  // TransposeMatMulFold stamps these with getI64IntegerAttr, i.e. signless i64,
  // so read through getValue(); getSInt() asserts on a non-signed type.
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transA"))
    transA = attr.getValue().getSExtValue();
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transB"))
    transB = attr.getValue().getSExtValue();

  // MatMul broadcasts leading axes and removes the promoted M/N dimension
  // for a vector operand. Compute dynamic output extents from both inputs.
  llvm::SmallVector<mlir::Value> dynSizes;
  const int64_t rank = resultType.getRank();
  const auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
  const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
  const int64_t aRank = aType.getRank();
  const int64_t bRank = bType.getRank();
  const int64_t aBatchRank = std::max<int64_t>(aRank - 2, 0);
  const int64_t bBatchRank = std::max<int64_t>(bRank - 2, 0);
  const int64_t batchRank = std::max(aBatchRank, bBatchRank);
  auto extent = [&](Value input, int64_t axis) -> Value {
    if (axis < 0)
      return arith::ConstantIndexOp::create(rewriter, loc, 1);
    return tensor::DimOp::create(rewriter, loc, input, axis);
  };
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx < batchRank) {
      const int64_t aAxis = dimIdx - (batchRank - aBatchRank);
      const int64_t bAxis = dimIdx - (batchRank - bBatchRank);
      const int64_t aSize = aAxis < 0 ? 1 : aType.getDimSize(aAxis);
      const int64_t bSize = bAxis < 0 ? 1 : bType.getDimSize(bAxis);
      // A missing or unit axis broadcasts and a static non-unit extent fixes
      // the output; only two dynamic extents need a runtime choice.
      if (aSize == 1 || (!ShapedType::isDynamic(bSize) && bSize != 1)) {
        dynSizes.push_back(extent(b, bAxis));
        continue;
      }
      if (bSize == 1 || !ShapedType::isDynamic(aSize)) {
        dynSizes.push_back(extent(a, aAxis));
        continue;
      }
      Value ad = extent(a, aAxis);
      Value bd = extent(b, bAxis);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value aIsOne = arith::CmpIOp::create(rewriter, loc,
                                           arith::CmpIPredicate::eq, ad, one);
      dynSizes.push_back(
          arith::SelectOp::create(rewriter, loc, aIsOne, bd, ad));
    } else if (aRank > 1 && dimIdx == batchRank) {
      int64_t aDim = transA ? aRank - 1 : aRank - 2;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, aDim));
    } else {
      int64_t bDim = transB ? bRank - 2 : bRank - 1;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, b, bDim));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(transA)));
  attrs.push_back(
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)));
  llvm::SmallVector<mlir::Value> operands = {context, a, b, init};
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
