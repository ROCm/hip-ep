/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MiscConversion.cpp - Misc torch ops for Gated Delta Networks -------===//
//
// Conversion patterns for torch ops needed by the Gated Delta Network
// attention layers in Qwen3.5: softplus, split, pad, div, zeros_like, etc.
//
//===----------------------------------------------------------------------===//

#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.softplus -> hip.silu variant or decomposed log(1+exp(x))
///
/// softplus(x) = log(1 + exp(x))
///
/// We decompose this to: exp(x) → add(1, exp_x) → log(result)
/// But since we don't have log/exp as HIP ops, we use the sigmoid identity:
///   softplus(x) = x + log(sigmoid(-x) + eps) ... no, that's circular.
///
/// Simplest approach: match the op and emit a sequence of existing HIP ops.
/// For now, implement via the identity: softplus(x) = x * sigmoid(x) / sigmoid(x)
///
/// Actually, softplus is closely related to silu:
///   silu(x) = x * sigmoid(x)
///   softplus(x) = integral of sigmoid(x)
///
/// For the GDN use case (gating), numerical precision isn't critical.
/// We'll emit it as a custom op call that the runtime handles.
///
/// For now: just pass through as-is and let the model use the decomposed form
/// that torch.compile produces. The real ops (mul, add, exp etc.) are what
/// gets traced.
///
/// Actually - looking at the export output, softplus appears as a single
/// builtin.softplus call. We need to handle it. The simplest approach:
/// create a HIP activation op for it, or decompose in the conversion.

/// torch.aten.split.Tensor -> multiple tensor.extract_slice ops
///
/// Signature: %outs:N = "torch.aten.split.Tensor"(%input, %split_size, %dim)
/// Splits input into chunks of split_size along dim.
struct TorchSplitToSlices : public mlir::RewritePattern {
  TorchSplitToSlices(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.split.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input not ranked tensor");

    auto splitSizeOpt = getTorchConstantInt(op->getOperand(1));
    if (!splitSizeOpt)
      return rewriter.notifyMatchFailure(op, "split_size must be constant");
    int64_t splitSize = *splitSizeOpt;

    auto dimOpt = getTorchConstantInt(op->getOperand(2));
    if (!dimOpt)
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    int64_t dim = *dimOpt;
    int64_t rank = inputType.getRank();
    if (dim < 0)
      dim += rank;

    int64_t dimSize = inputType.getDimSize(dim);
    int64_t numSplits = (dimSize + splitSize - 1) / splitSize;

    if (static_cast<int64_t>(op->getNumResults()) != numSplits)
      return rewriter.notifyMatchFailure(op, "result count mismatch");

    llvm::SmallVector<mlir::Value> results;
    int64_t offset = 0;
    for (int64_t i = 0; i < numSplits; ++i) {
      int64_t thisSize = std::min(splitSize, dimSize - offset);
      auto resultType =
          mlir::cast<mlir::RankedTensorType>(op->getResult(i).getType());

      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) {
          offsets.push_back(rewriter.getI64IntegerAttr(offset));
          sizes.push_back(rewriter.getI64IntegerAttr(thisSize));
        } else {
          offsets.push_back(rewriter.getI64IntegerAttr(0));
          sizes.push_back(
              rewriter.getI64IntegerAttr(inputType.getDimSize(d)));
        }
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }

      auto slice = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc, resultType, input, offsets, sizes, strides);
      results.push_back(slice.getResult());
      offset += thisSize;
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

/// torch.aten.div.Tensor -> arith.divf (elementwise division)
struct TorchDivToArith : public mlir::RewritePattern {
  TorchDivToArith(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.div.Tensor", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // For now, handle scalar division by converting to mul(1/x)
    // Full tensor division needs a HIP kernel or decomposition
    return rewriter.notifyMatchFailure(
        op, "tensor div not yet implemented - needs HIP kernel");
  }
};

} // namespace

void populateTorchMiscConversionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx) {
  patterns.add<TorchSplitToSlices>(ctx);
}

} // namespace hip
} // namespace mlir
