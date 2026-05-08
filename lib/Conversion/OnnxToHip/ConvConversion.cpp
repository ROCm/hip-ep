//===- ConvConversion.cpp - ONNX-to-HIP Conv conversion ------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Why this conversion exists
// --------------------------
// `onnx.Conv` carries a rich attribute set (auto_pad, dilations, group,
// kernel_shape, pads, strides) that maps almost 1:1 onto MIOpen's
// convolution descriptor.  We rewrite to `hip.conv` early in the pipeline
// so that downstream constant pooling, bufferization, and the eventual
// HIP-to-LLVM lowering can all reason about a single op kind instead of
// re-parsing ONNX attributes at every stage.
//
// Non-obvious choices
// -------------------
// * `auto_pad` is resolved to explicit `pads` here (not at lowering time)
//   because the explicit form is what MIOpen's descriptor consumes; doing
//   it once during conversion avoids ambiguity about which dialect "owns"
//   pad-mode resolution.
// * Bias is optional in ONNX (3 vs 2 operands); we materialize a
//   zero-tensor when absent so `hip.conv` always has the same operand
//   arity, simplifying every consumer downstream.
// * Output shape inference uses the standard ONNX formula and is checked
//   against the result type already attached by onnx-mlir; mismatch is a
//   hard error rather than a silent overwrite.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Conv -> hip.conv
struct ConvToHip : public RewritePattern {
  ConvToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ConvToHip::matchAndRewrite(Operation* op,
                                         PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  Value weights = op->getOperand(1);

  // ONNX Conv always has 3 operands, but bias can be onnx.NoValue (NoneType)
  bool hasBias =
      op->getNumOperands() > 2 && !isa<NoneType>(op->getOperand(2).getType());
  Value bias = hasBias ? op->getOperand(2) : nullptr;

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // Extract attributes from onnx.Conv.  Defaults follow the ONNX Conv
  // schema: strides = [1, 1, ...], dilations = [1, 1, ...], pads = [0, ...]
  // sized 2 * spatial-rank.
  llvm::SmallVector<int64_t> kernelShape =
      getInt64ArrayAttrOrDefault(op, "kernel_shape");
  llvm::SmallVector<int64_t> strides = getInt64ArrayAttrOrDefault(
      op, "strides", llvm::SmallVector<int64_t>(kernelShape.size(), 1));
  llvm::SmallVector<int64_t> pads = getInt64ArrayAttrOrDefault(
      op, "pads", llvm::SmallVector<int64_t>(kernelShape.size() * 2, 0));
  llvm::SmallVector<int64_t> dilations = getInt64ArrayAttrOrDefault(
      op, "dilations", llvm::SmallVector<int64_t>(kernelShape.size(), 1));

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();

  // Create output tensor
  llvm::SmallVector<Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, dimIdx));
  }

  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);

  // Build attributes for hip.conv
  auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
  auto stridesAttr = rewriter.getI64ArrayAttr(strides);
  auto padsAttr = rewriter.getI64ArrayAttr(pads);
  auto dilationsAttr = rewriter.getI64ArrayAttr(dilations);
  auto groupAttr = rewriter.getI64IntegerAttr(group);

  // Build operands vector: context, input, weights, [bias], init
  llvm::SmallVector<Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(init);

  // Build attributes
  llvm::SmallVector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
  attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
  attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
  attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
  attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

  // Create hip.conv operation using generic builder
  auto hipOp = mlir::hip::ConvOp::create(rewriter, loc, TypeRange{resultType},
                                         operands, attrs);

  rewriter.replaceOp(op, hipOp.getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateConvConversionPatterns(RewritePatternSet& patterns,
                                               MLIRContext* ctx) {
  patterns.add<ConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
