/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "ReadbackScalar.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

/// onnx.CumSum -> hip.cumsum
struct CumSumToHip : public mlir::RewritePattern {
  CumSumToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.CumSum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    mlir::Value axis = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Output shape mirrors the data input shape.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

    int64_t exclusive = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("exclusive"))
      exclusive = attr.getValue().getSExtValue();
    int64_t reverse = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("reverse"))
      reverse = attr.getValue().getSExtValue();

    std::optional<int64_t> axisConst;
    if (mlir::DenseElementsAttr dense = getConstantDense(axis)) {
      auto elemTy = dense.getElementType();
      if (dense.getNumElements() == 1 &&
          (elemTy.isInteger(32) || elemTy.isInteger(64))) {
        int64_t value =
            (*dense.getValues<llvm::APInt>().begin()).getSExtValue();
        int64_t rank = resultType.getRank();
        if (value < 0)
          value += rank;
        if (value >= 0 && value < rank)
          axisConst = value;
      }
    }

    mlir::SmallVector<mlir::Value> operands{context, x};
    if (!axisConst)
      operands.push_back(axis);
    operands.push_back(init);

    mlir::OperationState state(loc, "hip.cumsum");
    state.addOperands(operands);
    state.addTypes({resultType});
    if (axisConst)
      state.addAttribute("axis_attr", rewriter.getI64IntegerAttr(*axisConst));
    if (exclusive)
      state.addAttribute("exclusive", rewriter.getI64IntegerAttr(exclusive));
    if (reverse)
      state.addAttribute("reverse", rewriter.getI64IntegerAttr(reverse));

    mlir::Operation *hipOp = rewriter.create(state);
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populateCumSumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<CumSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
