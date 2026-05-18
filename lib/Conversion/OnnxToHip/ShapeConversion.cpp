/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shape -> arith.constant (compile-time fold)
//===----------------------------------------------------------------------===//
//
// The MorphiZen MLIR compiler requires fully static input shapes (see
// CLAUDE.md). With static shapes, `onnx.Shape(input)` is a pure compile-time
// constant: the output is just the input's dimension sizes packed into a
// rank-1 int64 tensor.  We therefore fold it to an `arith.constant` directly,
// matching the form produced by `lowerOnnxConstants` for ordinary
// `onnx.Constant` ops.  No runtime support is needed.
//
// ONNX Shape spec (opset >= 15) supports two optional integer attributes
// `start` and `end` that slice the rank dimensions (similar to Python list
// slicing); negative values count from the back and out-of-range values are
// clamped.  We replicate those semantics here so that the fold is correct
// regardless of how the source model emits the op.
struct ShapeFold : public mlir::RewritePattern {
  ShapeFold(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    auto input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor input");

    int64_t rank = inputType.getRank();

    // Resolve start: default 0; negative offsets from rank; clamp into
    // [0, rank].  Spec: "specifying any start value < -r is equivalent to
    // specifying a start value of 0".
    int64_t start = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("start"))
      start = attr.getValue().getSExtValue();
    if (start < 0)
      start += rank;
    start = std::clamp<int64_t>(start, 0, rank);

    // Resolve end: default rank (means "all the way to the last axis");
    // negative offsets from rank; clamp into [0, rank].  Spec: "specifying
    // any end value > r is equivalent to specifying an end value of r".
    int64_t end = rank;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("end"))
      end = attr.getValue().getSExtValue();
    if (end < 0)
      end += rank;
    end = std::clamp<int64_t>(end, 0, rank);

    // Spec: "If start > end, the result will be an empty shape."
    if (start > end)
      end = start;

    int64_t outLen = end - start;

    // Static-shape requirement: every dim in the slice must be known.  If any
    // is dynamic we cannot fold, so bail out and let upstream see the failure.
    // In production this should never trigger because the EP rejects dynamic
    // shape models up front; the check exists so unit tests with dynamic dims
    // produce a clear notifyMatchFailure rather than incorrect values.
    llvm::SmallVector<int64_t> shapeValues;
    shapeValues.reserve(outLen);
    for (int64_t i = start; i < end; ++i) {
      int64_t dim = inputType.getDimSize(i);
      if (dim == mlir::ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "onnx.Shape fold requires static input dim");
      shapeValues.push_back(dim);
    }

    // Build the resulting rank-1 int64 tensor and the corresponding
    // arith.constant.  Prefer the op's actual result type when its shape and
    // element type match our computed values, so downstream type matching is
    // exact; otherwise synthesize a fresh type.
    auto outType =
        mlir::RankedTensorType::get({outLen}, rewriter.getI64Type());
    auto opResultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (opResultType && opResultType.getRank() == 1 &&
        opResultType.getElementType().isInteger(64) &&
        !opResultType.isDynamicDim(0) &&
        opResultType.getDimSize(0) == outLen)
      outType = opResultType;

    auto denseAttr = mlir::DenseElementsAttr::get(
        outType, llvm::ArrayRef<int64_t>(shapeValues));
    mlir::Value cst =
        mlir::arith::ConstantOp::create(rewriter, op->getLoc(), denseAttr);
    rewriter.replaceOp(op, cst);
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateShapeConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<ShapeFold>(ctx);
}

} // namespace hip
} // namespace mlir
