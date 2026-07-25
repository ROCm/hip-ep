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
// Size -> arith.constant (static) OR hip.size (dynamic)
//===----------------------------------------------------------------------===//
//
// ONNX `Size` produces a rank-0 int64 tensor whose single element is the
// total number of elements of the input tensor (product of all input
// dimensions).
//
// Two paths:
//
//  1. Static input shape -- the historical fast path. `prod(input.shape)`
//     is a compile-time integer, so we collapse `onnx.Size` into a single
//     `arith.constant`. No HIP op, no runtime function, no per-inference
//     work. The input value itself becomes dead and is cleaned up by
//     later DCE. This is what every model produced via `fix_shapes()`
//     hits today.
//
//  2. Dynamic input shape -- at least one dim is `?`. We emit
//     `hip.size(%ctx) ins(%x) outs(%init : tensor<i64>)`. The HipToLLVM
//     lowering uses `computeNumElements` (compile-time const for static
//     dims, descriptor `sizes[i]` for dynamic dims) to build the i64
//     product at runtime and stores it into the rank-0 output buffer.
//
// Keeping the static path identical to before is intentional -- the e2e
// lit test (test/lit/e2e/test_size_model.mlir) CHECK-NOTs `onnx.Size`
// and does NOT require any `wrap_*` symbol, and that contract MUST hold
// because pre-existing models bake the constant into the DLL.

struct SizeToConstantOrHipSize : public mlir::RewritePattern {
  SizeToConstantOrHipSize(mlir::MLIRContext *ctx)
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

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    // ONNX Size always produces a rank-0 int64 tensor. Other element types
    // would indicate a malformed graph.
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must have i64 elements");
    if (resultType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "result must be rank-0");

    mlir::Location loc = op->getLoc();

    if (inputType.hasStaticShape()) {
      // Path 1: compile-time fold to arith.constant.
      int64_t numElements = 1;
      for (int64_t d : inputType.getShape())
        numElements *= d;

      auto i64Ty = rewriter.getIntegerType(64);
      auto scalarTensorTy = mlir::RankedTensorType::get({}, i64Ty);
      auto attr = mlir::DenseElementsAttr::get(
          scalarTensorTy, mlir::APInt(64, numElements, /*isSigned=*/true));

      mlir::Value cst = mlir::arith::ConstantOp::create(rewriter, loc, attr);
      rewriter.replaceOp(op, cst);
      return mlir::success();
    }

    // Path 2: dynamic input shape. Emit hip.size in destination-passing
    // style. The output is a rank-0 i64 tensor; `tensor.empty()` takes
    // no dyn-size operands because the result has no dynamic dims.
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{});

    auto hipOp = mlir::hip::SizeOp::create(rewriter, loc, context,
                                           op->getOperand(0), init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSizeConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<SizeToConstantOrHipSize>(ctx);
}

} // namespace hip
} // namespace mlir
