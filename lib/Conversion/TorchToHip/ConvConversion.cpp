/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.conv2d -> hip.conv
struct TorchConv2dToHip : public mlir::RewritePattern {
  TorchConv2dToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.conv2d", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Operands: (input, weight, bias, stride, padding, dilation, groups)
    mlir::Value input = op->getOperand(0);
    mlir::Value weight = op->getOperand(1);
    mlir::Value biasVal = op->getOperand(2);

    bool hasBias = !isTorchNone(biasVal);
    mlir::Value bias = hasBias ? biasVal : nullptr;

    // Extract stride (operand 3) - list of 2 ints
    auto strideOpt = getTorchConstantIntList(op->getOperand(3));
    if (!strideOpt || strideOpt->size() != 2)
      return rewriter.notifyMatchFailure(op, "stride must be a list of 2 ints");

    // Extract padding (operand 4) - list of 2 ints
    auto paddingOpt = getTorchConstantIntList(op->getOperand(4));
    if (!paddingOpt || paddingOpt->size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "padding must be a list of 2 ints");

    // Extract dilation (operand 5) - list of 2 ints
    auto dilationOpt = getTorchConstantIntList(op->getOperand(5));
    if (!dilationOpt || dilationOpt->size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "dilation must be a list of 2 ints");

    // Extract groups (operand 6) - single int
    auto groupsOpt = getTorchConstantInt(op->getOperand(6));
    if (!groupsOpt)
      return rewriter.notifyMatchFailure(op, "groups must be a constant int");

    // Extract kernel_shape from weight tensor type (dims 2, 3)
    auto weightType = mlir::cast<mlir::RankedTensorType>(weight.getType());
    if (weightType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "weight must be a 4D tensor");
    llvm::SmallVector<int64_t> kernelShape = {weightType.getDimSize(2),
                                              weightType.getDimSize(3)};

    // Convert padding from [ph, pw] to [ph, pw, ph, pw] for HIP
    llvm::SmallVector<int64_t> pads = {(*paddingOpt)[0], (*paddingOpt)[1],
                                       (*paddingOpt)[0], (*paddingOpt)[1]};

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Create output tensor
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
      if (resultType.isDynamicDim(dimIdx))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx));
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    // Build attributes for hip.conv
    auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
    auto stridesAttr = rewriter.getI64ArrayAttr(*strideOpt);
    auto padsAttr = rewriter.getI64ArrayAttr(pads);
    auto dilationsAttr = rewriter.getI64ArrayAttr(*dilationOpt);
    auto groupAttr = rewriter.getI64IntegerAttr(*groupsOpt);

    // Build operands vector: context, input, weights, [bias], init
    llvm::SmallVector<mlir::Value> operands = {context, input, weight};
    if (bias)
      operands.push_back(bias);
    operands.push_back(init);

    // Build attributes
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
    attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
    attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
    attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
    attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

    // Create hip.conv operation using generic builder
    auto hipOp = mlir::hip::ConvOp::create(
        rewriter, loc, mlir::TypeRange{resultType}, operands, attrs);

    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchConvConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<TorchConv2dToHip>(ctx);
}

} // namespace hip
} // namespace mlir
