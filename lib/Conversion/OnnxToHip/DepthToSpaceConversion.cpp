/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hip {
namespace {

// onnx.DepthToSpace -> expand_shape + hip.transpose + collapse_shape.
//
// ONNX defines the op on NCHW only. `blocksize` (B) must be >= 1, and the
// channel count must be divisible by B^2. The output is
// [N, C / B^2, H * B, W * B]. `mode` selects which way the channel axis is
// split before the permutation (default DCR):
//
//   DCR:
//     Before:
//       %y = onnx.DepthToSpace %x {blocksize = B, mode = "DCR"}
//              : tensor<NxCxHxW> -> tensor<NxCoutxH*BxW*B>
//     After:
//       %e = tensor.expand_shape %x [[0], [1, 2, 3], [4], [5]]
//              output_shape [N, B, B, Cout, H, W]
//       %t = hip.transpose %e {perm = [0, 3, 4, 1, 5, 2]}
//              : tensor<NxCoutxHxBxWxB>
//       %y = tensor.collapse_shape %t [[0], [1], [2, 3], [4, 5]]
//
//   CRD is the same collapse, with the channel split [Cout, B, B] and
//   perm = [0, 1, 4, 2, 5, 3].
//
// B == 1 only inserts and removes size-1 axes, so both modes are the identity
// and the input is forwarded. Element types are limited to the 8/16/32/64-bit
// integers and floats hip.transpose lowers.
//
// A dim that is dynamic on the input but static on the result is first
// pinned on the input with a tensor.cast. expand_shape cannot split a dynamic
// source dim into an all-static group, and collapse_shape cannot produce a
// static dim from dynamic ones, so the reshapes must see the refined type.

struct DepthToSpaceDecompose : public mlir::RewritePattern {
  DepthToSpaceDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DepthToSpace", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
DepthToSpaceDecompose::matchAndRewrite(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  if (inputType.getRank() != 4 || outputType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "DepthToSpace requires rank-4 NCHW input and output");
  if (inputType.getElementType() != outputType.getElementType())
    return rewriter.notifyMatchFailure(op, "element type mismatch");

  mlir::Type elemType = inputType.getElementType();
  if (!elemType.isIntOrFloat())
    return rewriter.notifyMatchFailure(
        op, "element type must be an integer or a float");
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32 && bitWidth != 64)
    return rewriter.notifyMatchFailure(
        op, "element type must be 8, 16, 32, or 64 bits");

  auto blocksizeAttr = op->getAttrOfType<mlir::IntegerAttr>("blocksize");
  if (!blocksizeAttr)
    return rewriter.notifyMatchFailure(op, "missing blocksize");
  int64_t blocksize = blocksizeAttr.getValue().getSExtValue();
  if (blocksize < 1)
    return rewriter.notifyMatchFailure(op, "blocksize must be >= 1");
  int64_t bs2 = 0;
  if (llvm::MulOverflow(blocksize, blocksize, bs2))
    return rewriter.notifyMatchFailure(op, "blocksize^2 overflows int64");

  llvm::StringRef mode = "DCR";
  if (auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode"))
    mode = modeAttr.getValue();
  const bool isDCR = mode == "DCR";
  if (!isDCR && mode != "CRD")
    return rewriter.notifyMatchFailure(op, "mode must be DCR or CRD");

  const int64_t dyn = mlir::ShapedType::kDynamic;

  // Per axis, the input extent multiplied by `scale` must equal the output
  // extent (channels scale by 1 / B^2, spatial dims by B). Where only one
  // side is static, the other is derived from it and must be integral.
  // `refined` collects the input shape with every provable extent filled in.
  llvm::SmallVector<int64_t> refined(inputType.getShape());
  auto checkAxis = [&](int64_t axis, int64_t mul,
                       int64_t div) -> mlir::LogicalResult {
    int64_t in = inputType.getDimSize(axis);
    int64_t out = outputType.getDimSize(axis);
    if (in != dyn) {
      if (in % div != 0)
        return rewriter.notifyMatchFailure(
            op, "input channels must be divisible by blocksize^2");
      int64_t expected = 0;
      if (llvm::MulOverflow(in / div, mul, expected))
        return rewriter.notifyMatchFailure(op, "output extent overflows int64");
      if (out != dyn && out != expected)
        return rewriter.notifyMatchFailure(
            op, "output shape must be [N, C / blocksize^2, H * blocksize, "
                "W * blocksize]");
      return mlir::success();
    }
    if (out == dyn)
      return mlir::success();
    if (out % mul != 0)
      return rewriter.notifyMatchFailure(
          op, "static output spatial size must be divisible by blocksize");
    if (llvm::MulOverflow(out / mul, div, refined[axis]))
      return rewriter.notifyMatchFailure(op, "input extent overflows int64");
    return mlir::success();
  };
  if (mlir::failed(checkAxis(/*axis=*/0, /*mul=*/1, /*div=*/1)) ||
      mlir::failed(checkAxis(/*axis=*/1, /*mul=*/1, /*div=*/bs2)) ||
      mlir::failed(checkAxis(/*axis=*/2, /*mul=*/blocksize, /*div=*/1)) ||
      mlir::failed(checkAxis(/*axis=*/3, /*mul=*/blocksize, /*div=*/1)))
    return mlir::failure();

  // blocksize 1 does not reorder elements in either mode.
  if (blocksize == 1) {
    if (inputType == outputType)
      rewriter.replaceOp(op, input);
    else
      rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, outputType, input);
    return mlir::success();
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();

  mlir::Location loc = op->getLoc();
  auto refinedType = mlir::RankedTensorType::get(refined, elemType);
  if (refinedType != inputType)
    input = mlir::tensor::CastOp::create(rewriter, loc, refinedType, input);

  auto extentOf = [&](int64_t axis) -> mlir::OpFoldResult {
    if (refined[axis] != dyn)
      return rewriter.getIndexAttr(refined[axis]);
    mlir::Value dim = mlir::tensor::DimOp::create(rewriter, loc, input, axis);
    return dim;
  };
  mlir::OpFoldResult nFold = extentOf(0);
  mlir::OpFoldResult hFold = extentOf(2);
  mlir::OpFoldResult wFold = extentOf(3);
  mlir::OpFoldResult blockFold = rewriter.getIndexAttr(blocksize);

  int64_t coutShape = dyn;
  mlir::OpFoldResult coutFold;
  if (refined[1] != dyn) {
    coutShape = refined[1] / bs2;
    coutFold = rewriter.getIndexAttr(coutShape);
  } else {
    mlir::Value channels =
        mlir::tensor::DimOp::create(rewriter, loc, input, /*dim=*/1);
    mlir::Value divisor =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, bs2);
    mlir::Value quotient =
        mlir::arith::DivSIOp::create(rewriter, loc, channels, divisor);
    coutFold = quotient;
  }
  const int64_t nShape = refined[0];
  const int64_t hShape = refined[2];
  const int64_t wShape = refined[3];

  llvm::SmallVector<int64_t> expandedShape;
  llvm::SmallVector<mlir::OpFoldResult> expandedFold;
  if (isDCR) {
    expandedShape = {nShape, blocksize, blocksize, coutShape, hShape, wShape};
    expandedFold = {nFold, blockFold, blockFold, coutFold, hFold, wFold};
  } else {
    expandedShape = {nShape, coutShape, blocksize, blocksize, hShape, wShape};
    expandedFold = {nFold, coutFold, blockFold, blockFold, hFold, wFold};
  }
  auto expandedType = mlir::RankedTensorType::get(expandedShape, elemType);
  llvm::SmallVector<mlir::ReassociationIndices> expandReassoc = {
      {0}, {1, 2, 3}, {4}, {5}};
  mlir::Value expanded = mlir::tensor::ExpandShapeOp::create(
      rewriter, loc, expandedType, input, expandReassoc, expandedFold);

  llvm::SmallVector<int64_t> transposedShape = {
      nShape, coutShape, hShape, blocksize, wShape, blocksize};
  llvm::SmallVector<mlir::Value> transposedDyn;
  for (mlir::OpFoldResult extent : {nFold, coutFold, hFold, wFold})
    if (auto value = llvm::dyn_cast<mlir::Value>(extent))
      transposedDyn.push_back(value);
  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, transposedShape, elemType, transposedDyn);

  llvm::SmallVector<int64_t> perm =
      isDCR ? llvm::SmallVector<int64_t>{0, 3, 4, 1, 5, 2}
            : llvm::SmallVector<int64_t>{0, 1, 4, 2, 5, 3};
  auto transposed =
      mlir::hip::TransposeOp::create(rewriter, loc, *ctxOrFailure, expanded,
                                     init, rewriter.getI64ArrayAttr(perm));

  auto scaled = [&](int64_t extent) {
    return extent == dyn ? dyn : extent * blocksize;
  };
  auto collapsedType = mlir::RankedTensorType::get(
      {nShape, coutShape, scaled(hShape), scaled(wShape)}, elemType);
  llvm::SmallVector<mlir::ReassociationIndices> collapseReassoc = {
      {0}, {1}, {2, 3}, {4, 5}};
  mlir::Value result = mlir::tensor::CollapseShapeOp::create(
      rewriter, loc, collapsedType, transposed->getResult(0), collapseReassoc);
  if (collapsedType != outputType)
    result = mlir::tensor::CastOp::create(rewriter, loc, outputType, result);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

} // namespace

void populateDepthToSpaceConversionPatterns(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  patterns.add<DepthToSpaceDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
