/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Size -> arith.constant (compile-time fold)
//===----------------------------------------------------------------------===//
//
// ONNX `Size` produces a rank-0 int64 tensor whose single element is the
// total number of elements of the input tensor (product of all input
// dimensions).
//
// The MLIR compiler in this EP requires fully static input shapes (see
// CLAUDE.md "Models must have static shapes"), so for every input that
// reaches this pattern the element count is a compile-time integer. That
// means `Size` always collapses to a single `arith.constant` -- no HIP op,
// no runtime function, no per-inference work. The input value itself
// becomes dead and is cleaned up by later DCE.
//
// If the input ever has dynamic dims we bail out via `notifyMatchFailure`
// so that any future dyn-shape extension surfaces the gap loudly instead
// of silently producing a wrong constant.

struct SizeToConstant : public mlir::RewritePattern {
  SizeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Size", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Size requires static input shape (this EP rejects dynamic "
              "shapes)");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    // ONNX Size always produces a rank-0 int64 tensor. Other element types
    // would indicate a malformed graph.
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must have i64 elements");

    // Product of all dims = number of elements. Static-shape guarantee
    // above means every dim is a fixed positive integer.
    int64_t numElements = 1;
    for (int64_t d : inputType.getShape())
      numElements *= d;

    auto i64Ty = rewriter.getIntegerType(64);
    auto scalarTensorTy = mlir::RankedTensorType::get({}, i64Ty);
    auto attr = mlir::DenseElementsAttr::get(
        scalarTensorTy, mlir::APInt(64, numElements, /*isSigned=*/true));

    mlir::Value cst =
        mlir::arith::ConstantOp::create(rewriter, op->getLoc(), attr);
    rewriter.replaceOp(op, cst);
    return mlir::success();
  }
};

} // namespace

void populateSizeConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<SizeToConstant>(ctx);
}

} // namespace hip
} // namespace mlir
