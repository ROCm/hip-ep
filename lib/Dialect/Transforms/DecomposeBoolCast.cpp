/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Rewrite a non-bool hip.cast to bool as x != 0.
//
// The cast kernels convert by value: an int64 5 or a float 2.5 would land in
// the one-byte mask as 5 or 2, and later Where/And/Not read it wrong. ONNX
// Cast and torch to(bool) both define the result as x != 0, which hip.equal
// and hip.not already compute for every type hip.equal supports.
//
// A cast to bool is one with an i1 result (torch) or `to = BOOL` (ONNX, whose
// importer stores BOOL as ui8, so the result type alone looks like UINT8).
//
// Before:
//   %y = hip.cast(%ctx) ins(%x : tensor<?x4xi64>) outs(%e : tensor<?x4xi1>)
//        {to = 9 : i64}
// After:
//   %z = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
//   %q = hip.equal(%ctx) ins(%x, %z : tensor<?x4xi64>, tensor<i64>)
//        outs(%e0 : tensor<?x4xi1>)
//   %y = hip.not(%ctx) ins(%q : tensor<?x4xi1>) outs(%e : tensor<?x4xi1>)
//
// A cast that also reshapes (tensor<1xT> to tensor<U>) computes the mask in
// the input shape and keeps a 0/1 cast for the reshape.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {
#define GEN_PASS_DEF_DECOMPOSEBOOLCASTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// ONNX TensorProto codes used as hip.cast `to`.
constexpr int64_t kOnnxFloat = 1;
constexpr int64_t kOnnxUint8 = 2;
constexpr int64_t kOnnxBool = 9;

// Element types the hip.equal runtime compares directly.
bool isEqualElementType(Type type) {
  if (type.isF16() || type.isF32())
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && llvm::is_contained({8u, 16u, 32u, 64u}, integer.getWidth());
}

Value emptyLike(PatternRewriter &rewriter, Location loc, Value shapeSource,
                Type elementType) {
  auto type = cast<RankedTensorType>(shapeSource.getType());
  SmallVector<Value> dynSizes;
  for (int64_t dim : llvm::seq<int64_t>(type.getRank()))
    if (type.isDynamicDim(dim))
      dynSizes.push_back(
          tensor::DimOp::create(rewriter, loc, shapeSource, dim));
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(), elementType,
                                 dynSizes);
}

struct DecomposeBoolCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "not a tensor-form cast");
    Type outputElement = outputType.getElementType();
    bool toBool = outputElement.isInteger(1) || op.getTo() == kOnnxBool;
    if (!toBool || inputType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "not a non-bool to bool cast");

    // hip.equal and hip.not take at most four dimensions. A static higher-rank
    // cast is flattened to rank 1 around the rewrite; a dynamic one is left
    // for hip-to-llvm to reject, which beats a mask that holds 5 instead of 1.
    bool packed = std::max(inputType.getRank(), outputType.getRank()) > 4;
    if (packed && (!inputType.hasStaticShape() ||
                   !outputType.hasStaticShape() || inputType.getRank() == 0))
      return rewriter.notifyMatchFailure(op, "unpackable shape of rank > 4");

    Type element = inputType.getElementType();
    if (!isEqualElementType(element) && !element.isBF16())
      return rewriter.notifyMatchFailure(op, "no hip.equal for this type");

    Location loc = op.getLoc();
    Value ctx = op.getCtx();
    Value input = op.getInput();
    if (packed) {
      auto flatType = RankedTensorType::get({inputType.getNumElements()},
                                            inputType.getElementType());
      ReassociationIndices all;
      for (int64_t dim : llvm::seq<int64_t>(inputType.getRank()))
        all.push_back(dim);
      input = tensor::CollapseShapeOp::create(rewriter, loc, flatType, input,
                                              ArrayRef{all});
    }
    // The shape source for the mask and its init, before any bf16 widening.
    Value shaped = input;
    if (element.isBF16()) {
      element = rewriter.getF32Type();
      Value init = emptyLike(rewriter, loc, input, element);
      input = CastOp::create(rewriter, loc, ctx, input, init,
                             rewriter.getI64IntegerAttr(kOnnxFloat))
                  ->getResult(0);
    }

    auto zeroType = RankedTensorType::get({}, element);
    Value zero = ConstantOp::create(
        rewriter, loc, zeroType,
        DenseElementsAttr::get(zeroType, rewriter.getZeroAttr(element)));
    // hip.equal broadcasts, so a tensor<1xT> input would keep its shape; the
    // mask is built in the input shape and reshaped by a 0/1 cast below.
    bool reshapes = inputType.getShape() != outputType.getShape();
    Value equalInit = emptyLike(rewriter, loc, shaped, outputElement);
    Value isZero = EqualOp::create(rewriter, loc, ctx, input, zero, equalInit)
                       ->getResult(0);
    if (!reshapes && !packed) {
      rewriter.replaceOpWithNewOp<NotOp>(op, ctx, isZero, op.getOutput());
      return success();
    }
    Value notInit = emptyLike(rewriter, loc, shaped, outputElement);
    Value mask =
        NotOp::create(rewriter, loc, ctx, isZero, notInit)->getResult(0);
    if (packed) {
      // A packed cast has the same element count in and out, so the flat mask
      // expands straight to the result shape.
      if (outputType.getRank() == 1) {
        rewriter.replaceOp(op, mask);
        return success();
      }
      ReassociationIndices all;
      for (int64_t dim : llvm::seq<int64_t>(outputType.getRank()))
        all.push_back(dim);
      SmallVector<OpFoldResult> sizes;
      for (int64_t dim : outputType.getShape())
        sizes.push_back(rewriter.getIndexAttr(dim));
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, outputType, mask,
                                                         ArrayRef{all}, sizes);
      return success();
    }
    // The mask is already 0/1. An i1 source is never rewritten again; a ui8
    // one would be if it kept `to = BOOL`, so it is cast as UINT8.
    int64_t to = outputElement.isInteger(1) ? kOnnxBool : kOnnxUint8;
    rewriter.replaceOpWithNewOp<CastOp>(op, ctx, mask, op.getOutput(),
                                        rewriter.getI64IntegerAttr(to));
    return success();
  }
};

struct DecomposeBoolCastPass
    : public impl::DecomposeBoolCastPassBase<DecomposeBoolCastPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeBoolCast>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
