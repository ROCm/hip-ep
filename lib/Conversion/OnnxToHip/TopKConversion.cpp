/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "ReadbackScalar.h"

namespace mlir {
namespace hip {
namespace {

static mlir::Value getTopKAxisExtent(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value ctx,
                                     mlir::Value k, int64_t axis,
                                     mlir::RankedTensorType resultType) {
  if (!resultType.isDynamicDim(axis)) {
    return mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                resultType.getDimSize(axis));
  }

  auto kType = mlir::cast<mlir::RankedTensorType>(k.getType());
  if (kType.getRank() == 0) {
    mlir::Value kScalar = readbackScalarToHost(rewriter, loc, ctx, k);
    return mlir::arith::IndexCastOp::create(rewriter, loc,
                                            rewriter.getIndexType(), kScalar);
  }

  if (kType.getRank() == 1) {
    if (kType.hasStaticShape() && kType.getDimSize(0) == 1) {
      mlir::Value collapsed = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, k, mlir::ReassociationIndices{{0}});
      mlir::Value kScalar = readbackScalarToHost(rewriter, loc, ctx, collapsed);
      return mlir::arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), kScalar);
    }
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    return mlir::tensor::DimOp::create(rewriter, loc, k, c0);
  }

  return mlir::Value{};
}

static mlir::Value createTopKEmpty(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc,
                                   mlir::RankedTensorType resultType,
                                   mlir::Value source, mlir::Value ctx,
                                   mlir::Value k, int64_t axis) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(i))
      continue;
    if (i == axis) {
      mlir::Value axisExtent =
          getTopKAxisExtent(rewriter, loc, ctx, k, axis, resultType);
      if (!axisExtent)
        return mlir::Value{};
      dynSizes.push_back(axisExtent);
    } else {
      mlir::Value dim = mlir::tensor::DimOp::create(rewriter, loc, source, i);
      dynSizes.push_back(dim);
    }
  }

  return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

/// onnx.TopK -> hip.top_k
struct TopKToHip : public mlir::RewritePattern {
  TopKToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.TopK", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TopKToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  if (op->getNumOperands() != 2 || op->getNumResults() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2 inputs and 2 outputs");

  mlir::Value x = op->getOperand(0);
  mlir::Value k = op->getOperand(1);
  auto xType = mlir::cast<mlir::RankedTensorType>(x.getType());

  int64_t axis = -1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();
  if (axis < 0)
    axis += xType.getRank();

  bool largest = true;
  if (auto largestAttr = op->getAttrOfType<mlir::BoolAttr>("largest"))
    largest = largestAttr.getValue();

  bool sorted = true;
  if (auto sortedAttr = op->getAttrOfType<mlir::BoolAttr>("sorted"))
    sorted = sortedAttr.getValue();

  // TopK's two results share an IDENTICAL shape; only the element type differs
  // (values inherit x's element type, indices are integer). Some exports drop
  // the shape/rank on one result -- commonly the values result when only the
  // indices result is a graph output -- leaving it as an unranked tensor. An
  // unchecked cast to RankedTensorType then yields a null-backed handle and
  // dereferencing it in createTopKEmpty segfaults. Instead, reconstruct the
  // missing ranked type from its ranked sibling (same shape, own element type)
  // so conversion still proceeds.
  //
  // Before (values result unranked):
  //   %v, %i = "onnx.TopK"(%x, %k) : (tensor<?x?x512xf16>, tensor<1xi64>)
  //                                    -> (tensor<*xf16>, tensor<?x?x22xi64>)
  // After (values type rebuilt from indices shape + x elem type):
  //   %v, %i = hip.top_k ... -> (tensor<?x?x22xf16>, tensor<?x?x22xi64>)
  mlir::Type valuesRawType = op->getResult(0).getType();
  mlir::Type indicesRawType = op->getResult(1).getType();
  auto valuesType = mlir::dyn_cast<mlir::RankedTensorType>(valuesRawType);
  auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(indicesRawType);

  auto elemTypeOr = [](mlir::Type t, mlir::Type fallback) -> mlir::Type {
    if (auto tt = mlir::dyn_cast<mlir::TensorType>(t))
      return tt.getElementType();
    return fallback;
  };
  if (!valuesType && indicesType)
    valuesType = mlir::RankedTensorType::get(
        indicesType.getShape(),
        elemTypeOr(valuesRawType, xType.getElementType()));
  if (!indicesType && valuesType)
    indicesType = mlir::RankedTensorType::get(
        valuesType.getShape(),
        elemTypeOr(indicesRawType, rewriter.getI64Type()));
  if (!valuesType || !indicesType)
    return rewriter.notifyMatchFailure(op,
                                       "TopK results must be ranked tensors");

  mlir::Value valuesInit =
      createTopKEmpty(rewriter, loc, valuesType, x, context, k, axis);
  if (!valuesInit)
    return rewriter.notifyMatchFailure(op, "failed to build values init");

  mlir::Value indicesInit =
      createTopKEmpty(rewriter, loc, indicesType, x, context, k, axis);
  if (!indicesInit)
    return rewriter.notifyMatchFailure(op, "failed to build indices init");

  // hip.top_k is multi-result (autoInfer=0): pass explicit result types.
  auto hipOp = mlir::hip::TopKOp::create(
      rewriter, loc, mlir::TypeRange{valuesType, indicesType}, context, x, k,
      valuesInit, indicesInit, rewriter.getI64IntegerAttr(axis),
      rewriter.getBoolAttr(largest), rewriter.getBoolAttr(sorted));

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateTopKConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<TopKToHip>(ctx);
}

} // namespace hip
} // namespace mlir
