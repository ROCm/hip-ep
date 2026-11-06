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
  if (auto largestAttr = op->getAttrOfType<mlir::IntegerAttr>("largest"))
    largest = largestAttr.getSInt() != 0;

  bool sorted = true;
  if (auto sortedAttr = op->getAttrOfType<mlir::IntegerAttr>("sorted"))
    sorted = sortedAttr.getSInt() != 0;

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

  // hip.top_k is multi-result and does not use the single-result inference
  // family, so pass explicit result types.
  auto hipOp = mlir::hip::TopKOp::create(
      rewriter, loc, mlir::TypeRange{valuesType, indicesType}, context, x, k,
      valuesInit, indicesInit, rewriter.getI64IntegerAttr(axis),
      rewriter.getBoolAttr(largest), rewriter.getBoolAttr(sorted));

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

// Drop the size-1 axis that top-k keeps. A leading unit axis merges into the
// next dimension; any other unit axis merges into the previous one.
// Before: tensor<1x1xi64>, axis 1, keepdims 0
// After:  tensor.collapse_shape -> tensor<1xi64>
static mlir::FailureOr<mlir::Value>
squeezeUnitAxis(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value indices, int64_t axis,
                mlir::RankedTensorType resultType) {
  auto sourceType = mlir::cast<mlir::RankedTensorType>(indices.getType());
  int64_t rank = sourceType.getRank();
  if (resultType.getRank() + 1 != rank || axis < 0 || axis >= rank)
    return mlir::failure();
  if (rank == 1) {
    auto shapeType =
        mlir::RankedTensorType::get({0}, rewriter.getI64Type());
    mlir::Value shape = mlir::tensor::FromElementsOp::create(
        rewriter, loc, shapeType, mlir::ValueRange{});
    return mlir::Value(mlir::tensor::ReshapeOp::create(
        rewriter, loc, resultType, indices, shape));
  }
  llvm::SmallVector<mlir::ReassociationIndices> groups;
  for (int64_t index : llvm::seq<int64_t>(0, rank)) {
    if (index == axis)
      continue;
    groups.push_back({index});
  }
  if (axis == 0)
    groups.front().insert(groups.front().begin(), 0);
  else
    groups[axis - 1].push_back(axis);
  return mlir::Value(mlir::tensor::CollapseShapeOp::create(
      rewriter, loc, resultType, indices, groups));
}

/// onnx.ArgMax -> hip.top_k with K = 1. Ties keep the first index, matching
/// select_last_index = 0. Integer data is cast to f32 first; that cast is
/// exact for magnitudes up to 2^24, which covers token ids.
struct ArgMaxToHip : public mlir::RewritePattern {
  ArgMaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ArgMax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ArgMaxToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected one input and one output");

  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !inputType.hasStaticShape() || !resultType ||
      !resultType.getElementType().isInteger(64))
    return rewriter.notifyMatchFailure(
        op, "ArgMax requires a static input and an i64 result");

  int64_t selectLast = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("select_last_index"))
    selectLast = attr.getValue().getSExtValue();
  if (selectLast != 0)
    return rewriter.notifyMatchFailure(
        op, "ArgMax select_last_index must be 0");

  int64_t keepdims = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = attr.getValue().getSExtValue();
  if (keepdims != 0 && keepdims != 1)
    return rewriter.notifyMatchFailure(op, "ArgMax keepdims must be 0 or 1");

  int64_t axis = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = attr.getValue().getSExtValue();
  if (axis < 0)
    axis += inputType.getRank();
  if (axis < 0 || axis >= inputType.getRank())
    return rewriter.notifyMatchFailure(op, "ArgMax axis is out of range");

  mlir::Location loc = op->getLoc();
  mlir::Type element = inputType.getElementType();
  mlir::Value data = input;
  if (element.isIntOrIndex()) {
    auto castType =
        mlir::RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
    mlir::Value castInit = createEmptyTensor(rewriter, loc, castType, input);
    auto cast = mlir::hip::CastOp::create(rewriter, loc, context, input,
                                          castInit,
                                          rewriter.getI64IntegerAttr(1));
    data = cast->getResult(0);
    element = rewriter.getF32Type();
  } else if (!mlir::isa<mlir::FloatType>(element)) {
    return rewriter.notifyMatchFailure(op, "ArgMax element type is unsupported");
  }

  llvm::SmallVector<int64_t> keptShape(inputType.getShape());
  keptShape[axis] = 1;
  auto valuesType = mlir::RankedTensorType::get(keptShape, element);
  auto indicesType =
      mlir::RankedTensorType::get(keptShape, rewriter.getI64Type());
  if (keepdims == 1 && indicesType != resultType)
    return rewriter.notifyMatchFailure(op, "ArgMax result shape does not match");

  mlir::Value one = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(1));
  auto kType = mlir::RankedTensorType::get({1}, rewriter.getI64Type());
  mlir::Value k = mlir::tensor::FromElementsOp::create(rewriter, loc, kType,
                                                       mlir::ValueRange{one});
  mlir::Value valuesInit = createEmptyTensor(rewriter, loc, valuesType, data);
  mlir::Value indicesInit =
      createEmptyTensor(rewriter, loc, indicesType, data);
  auto topk = mlir::hip::TopKOp::create(
      rewriter, loc, mlir::TypeRange{valuesType, indicesType}, context, data, k,
      valuesInit, indicesInit, rewriter.getI64IntegerAttr(axis),
      rewriter.getBoolAttr(true), rewriter.getBoolAttr(true));
  mlir::Value indices = topk->getResult(1);
  if (keepdims == 0) {
    mlir::FailureOr<mlir::Value> squeezed =
        squeezeUnitAxis(rewriter, loc, indices, axis, resultType);
    if (mlir::failed(squeezed))
      return rewriter.notifyMatchFailure(op, "ArgMax result shape does not match");
    indices = *squeezed;
  }
  rewriter.replaceOp(op, indices);
  return mlir::success();
}

} // namespace

void populateTopKConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<TopKToHip, ArgMaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
