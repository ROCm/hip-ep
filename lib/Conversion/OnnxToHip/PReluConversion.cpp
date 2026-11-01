/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PReluConversion.cpp - onnx.PRelu -> hip.prelu ----------------------===//
//
// Y = X >= 0 ? X : slope * X. slope is unidirectional-broadcastable onto X,
// so a per-channel slope (1xCx1x1) and a scalar slope share one kernel.
//
//   Before:
//     %y = "onnx.PRelu"(%x, %slope)
//          : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>)
//            -> tensor<1x64x112x112xf32>
//   After:
//     %init = tensor.empty() : tensor<1x64x112x112xf32>
//     %y = hip.prelu(%ctx) ins(%x, %slope : tensor<1x64x112x112xf32>,
//                                        tensor<1x64x1x1xf32>)
//                          outs(%init : tensor<1x64x112x112xf32>)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

constexpr int64_t kPReluMaxRank = 8;

bool isPReluFloat(mlir::Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

// Right-align slope onto x. A static slope dim must be 1 or equal the
// corresponding x dim. Dynamic dims are checked again at launch.
bool slopeBroadcastsOnto(mlir::RankedTensorType slope,
                         mlir::RankedTensorType x) {
  if (slope.getRank() > x.getRank())
    return false;
  int64_t offset = x.getRank() - slope.getRank();
  for (int64_t i : llvm::seq<int64_t>(slope.getRank())) {
    if (slope.isDynamicDim(i) || x.isDynamicDim(offset + i))
      continue;
    int64_t slopeDim = slope.getDimSize(i);
    int64_t xDim = x.getDimSize(offset + i);
    if (slopeDim != 1 && slopeDim != xDim)
      return false;
  }
  return true;
}

struct PReluToHip : public mlir::RewritePattern {
  PReluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.PRelu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.PRelu expects X, slope, and one result");

    mlir::Value x = op->getOperand(0);
    mlir::Value slope = op->getOperand(1);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto slopeType = mlir::dyn_cast<mlir::RankedTensorType>(slope.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!xType || !slopeType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    if (xType.getElementType() != slopeType.getElementType() ||
        xType.getElementType() != resultType.getElementType() ||
        !isPReluFloat(xType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "PRelu requires matching f16, bf16, f32, or f64 tensors");
    if (xType.getRank() > kPReluMaxRank ||
        xType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(
          op, "PRelu result rank must match X and be <= 8");
    for (int64_t i : llvm::seq<int64_t>(xType.getRank())) {
      if (xType.isDynamicDim(i) || resultType.isDynamicDim(i))
        continue;
      if (xType.getDimSize(i) != resultType.getDimSize(i))
        return rewriter.notifyMatchFailure(op,
                                           "PRelu result shape must match X");
    }
    if (!slopeBroadcastsOnto(slopeType, xType))
      return rewriter.notifyMatchFailure(
          op, "PRelu slope is not unidirectional-broadcastable onto X");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Location loc = op->getLoc();
    // The result shape is X. slope never contributes an extent.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);
    auto hipOp = mlir::hip::PReluOp::create(rewriter, loc, resultType,
                                            *ctxOrFailure, x, slope, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populatePReluConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<PReluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
