/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.non_max_suppression(ctx, boxes, scores, max, iou, score, selected, count)
//   -> wrap_non_max_suppression(state, boxes, scores, max, iou, score,
//                               selected, count, num_batches, num_classes,
//                               num_boxes, capacity, center_point_box,
//                               has_score_threshold)
struct NonMaxSuppressionOpLowering
    : public ConvertOpToLLVMPattern<NonMaxSuppressionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NonMaxSuppressionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto boxesType = dyn_cast<MemRefType>(op.getBoxes().getType());
    auto scoresType = dyn_cast<MemRefType>(op.getScores().getType());
    auto selectedType = dyn_cast<MemRefType>(op.getSelected().getType());
    if (!boxesType || !scoresType || !selectedType ||
        boxesType.getRank() != 3 || scoresType.getRank() != 3 ||
        selectedType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "hip.non_max_suppression lowering expects ranked memrefs");

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto ptr = [&](Value memref) {
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    Value numBatches =
        getMemRefDimSize(boxesType, 0, adaptor.getBoxes(), rewriter, loc);
    Value numBoxes =
        getMemRefDimSize(boxesType, 1, adaptor.getBoxes(), rewriter, loc);
    Value numClasses =
        getMemRefDimSize(scoresType, 1, adaptor.getScores(), rewriter, loc);
    Value capacity =
        getMemRefDimSize(selectedType, 0, adaptor.getSelected(), rewriter, loc);

    SmallVector<Type, 14> paramTypes(8, ptrType);
    paramTypes.append(6, i64Type);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNonMaxSuppression, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {
        adaptor.getCtx(),
        ptr(adaptor.getBoxes()),
        ptr(adaptor.getScores()),
        ptr(adaptor.getMaxOutputBoxesPerClass()),
        ptr(adaptor.getIouThreshold()),
        ptr(adaptor.getScoreThreshold()),
        ptr(adaptor.getSelected()),
        ptr(adaptor.getCount()),
        numBatches,
        numClasses,
        numBoxes,
        capacity,
        createI64Const(op.getCenterPointBox()),
        createI64Const(op.getHasScoreThreshold() ? 1 : 0)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNonMaxSuppressionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NonMaxSuppressionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
