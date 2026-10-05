/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

static bool isRandomNormalElemType(mlir::Type elemType) {
  return elemType.isF16() || elemType.isBF16() || elemType.isF32() ||
         elemType.isF64();
}

// ONNX TensorProto dtype values that RandomNormalLike may write.
static mlir::Type typeFromOnnxDtype(mlir::MLIRContext *ctx, int64_t dtype) {
  switch (dtype) {
  case 1: // FLOAT
    return mlir::Float32Type::get(ctx);
  case 10: // FLOAT16
    return mlir::Float16Type::get(ctx);
  case 11: // DOUBLE
    return mlir::Float64Type::get(ctx);
  case 16: // BFLOAT16
    return mlir::BFloat16Type::get(ctx);
  default:
    return nullptr;
  }
}

static mlir::FailureOr<mlir::RankedTensorType>
inferRandomNormalResultType(mlir::Operation *op,
                            mlir::RankedTensorType inputType) {
  if (auto ranked =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType()))
    return ranked;

  mlir::Type elemType = inputType.getElementType();
  if (auto dtypeAttr = op->getAttrOfType<mlir::IntegerAttr>("dtype")) {
    mlir::Type fromDtype =
        typeFromOnnxDtype(op->getContext(), dtypeAttr.getInt());
    if (!fromDtype)
      return mlir::failure();
    elemType = fromDtype;
  }
  if (!isRandomNormalElemType(elemType))
    return mlir::failure();
  return mlir::RankedTensorType::get(inputType.getShape(), elemType);
}

static mlir::Value createRandomNormalEmpty(mlir::PatternRewriter &rewriter,
                                           mlir::Location loc,
                                           mlir::RankedTensorType resultType,
                                           mlir::Value input) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i : llvm::seq<int64_t>(resultType.getRank())) {
    if (!resultType.isDynamicDim(i))
      continue;
    dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, i));
  }
  return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

/// onnx.RandomNormalLike -> hip.random_normal_like
struct RandomNormalLikeToHip : public mlir::RewritePattern {
  RandomNormalLikeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.RandomNormalLike", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
RandomNormalLikeToHip::matchAndRewrite(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(op, "input must be ranked");

  constexpr int64_t kMaxRank = 8;
  if (inputType.getRank() > kMaxRank)
    return rewriter.notifyMatchFailure(op, "rank must be in [0, 8]");

  auto resultTypeOr = inferRandomNormalResultType(op, inputType);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(op, "result type is not a float tensor");
  mlir::RankedTensorType resultType = *resultTypeOr;
  if (!isRandomNormalElemType(resultType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "output element type must be f16, bf16, f32, or f64");
  if (resultType.getRank() != inputType.getRank())
    return rewriter.notifyMatchFailure(op, "output rank must match the input");

  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (resultType.isDynamicDim(i) || inputType.isDynamicDim(i))
      continue;
    if (resultType.getDimSize(i) != inputType.getDimSize(i))
      return rewriter.notifyMatchFailure(
          op, "output shape must copy the input shape");
  }

  float mean = 0.0f;
  if (auto meanAttr = op->getAttrOfType<mlir::FloatAttr>("mean")) {
    if (!meanAttr.getValue().isFinite())
      return rewriter.notifyMatchFailure(op, "mean must be finite");
    mean = static_cast<float>(meanAttr.getValueAsDouble());
  }
  float scale = 1.0f;
  if (auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>("scale")) {
    if (!scaleAttr.getValue().isFinite())
      return rewriter.notifyMatchFailure(op, "scale must be finite");
    scale = static_cast<float>(scaleAttr.getValueAsDouble());
  }

  mlir::FloatAttr seedAttr;
  if (auto onnxSeed = op->getAttrOfType<mlir::FloatAttr>("seed")) {
    if (!onnxSeed.getValue().isFinite())
      return rewriter.notifyMatchFailure(op, "seed must be finite");
    seedAttr = rewriter.getF32FloatAttr(
        static_cast<float>(onnxSeed.getValueAsDouble()));
  }

  mlir::Value init = createRandomNormalEmpty(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::RandomNormalLikeOp::create(
      rewriter, loc, context, input, init, rewriter.getF32FloatAttr(mean),
      rewriter.getF32FloatAttr(scale), seedAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateRandomNormalLikeConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<RandomNormalLikeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
