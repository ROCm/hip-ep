/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Pad -> hip.pad
///
/// ONNX layout: Pad(data, pads, [constant_value], [axes]) {mode}.
/// Optional inputs may be present and typed `none` (onnx.NoValue) when omitted
/// by the producer.
struct PadToHip : public mlir::RewritePattern {
  PadToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pad", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value pads = op->getOperand(1);

    auto isNone = [](mlir::Value v) -> bool {
      return v && mlir::isa<mlir::NoneType>(v.getType());
    };

    mlir::Value constantValue = nullptr;
    if (op->getNumOperands() > 2 && !isNone(op->getOperand(2)))
      constantValue = op->getOperand(2);
    mlir::Value axes = nullptr;
    if (op->getNumOperands() > 3 && !isNone(op->getOperand(3)))
      axes = op->getOperand(3);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Trust ONNX shape inference for the output (potentially dynamic). Use
    // data as the fallback for any dynamic dim sources -- callers typically
    // run shape inference before this conversion.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);

    mlir::StringAttr modeAttr;
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("mode"))
      modeAttr = attr;
    else
      modeAttr = rewriter.getStringAttr("constant");

    // Build operands [ctx, data, pads, cval?, axes?, output] and segment
    // sizes for AttrSizedOperandSegments.
    mlir::SmallVector<mlir::Value> operands;
    operands.push_back(context);
    operands.push_back(data);
    operands.push_back(pads);
    if (constantValue)
      operands.push_back(constantValue);
    if (axes)
      operands.push_back(axes);
    operands.push_back(init);

    llvm::SmallVector<int32_t, 6> segmentSizes = {
        /*ctx=*/1,
        /*data=*/1,
        /*pads=*/1,
        /*constant_value=*/constantValue ? 1 : 0,
        /*axes=*/axes ? 1 : 0,
        /*output=*/1};

    mlir::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("mode", modeAttr));

    mlir::OperationState state(loc, "hip.pad");
    state.addOperands(operands);
    state.addAttributes(attrs);
    state.addTypes({resultType});
    state.addAttribute("operand_segment_sizes",
                       rewriter.getDenseI32ArrayAttr(segmentSizes));

    mlir::Operation *hipOp = rewriter.create(state);
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populatePadConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<PadToHip>(ctx);
}

} // namespace hip
} // namespace mlir
