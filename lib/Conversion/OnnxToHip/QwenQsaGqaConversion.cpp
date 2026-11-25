/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "OnnxToHipUtils.h"

namespace mlir::hip {
namespace {
struct QwenQsaGqaToHip : RewritePattern {
  QwenQsaGqaToHip(MLIRContext *ctx) : RewritePattern("onnx.Custom", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto name = op->getAttrOfType<StringAttr>("function_name");
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!name || name.getValue() != "QwenQsaGroupQueryAttention" || !domain ||
        domain.getValue() != "com.amd")
      return failure();
    if (op->getNumOperands() != 15 || op->getNumResults() != 3)
      return op->emitError(
          "QwenQsaGroupQueryAttention requires 15 inputs and 3 outputs");
    auto ctx = getContextArg(op, rewriter);
    if (failed(ctx))
      return failure();
    SmallVector<Value> operands{*ctx};
    llvm::append_range(operands, op->getOperands());
    for (unsigned i = 0; i < 3; ++i) {
      auto type = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!type)
        return op->emitError("Qwen QSA requires ranked output types");
      operands.push_back(createEmptyTensor(rewriter, op->getLoc(), type,
                                           op->getOperand(i == 0 ? 0 : i + 2)));
    }
    OperationState state(op->getLoc(), QwenQsaGqaOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    auto *qsa = rewriter.create(state);
    rewriter.replaceOp(op, qsa->getResults());
    return success();
  }
};
} // namespace
void populateQwenQsaGqaConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<QwenQsaGqaToHip>(ctx);
}
} // namespace mlir::hip
